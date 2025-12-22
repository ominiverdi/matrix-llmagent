"""Matrix client using matrix-nio for communication."""

import io
import logging
import os
from typing import Any

from nio import AsyncClient, AsyncClientConfig

logger = logging.getLogger(__name__)


class MatrixClient:
    """Matrix protocol client wrapper for matrix-nio."""

    def __init__(self, config: dict[str, Any]):
        """Initialize Matrix client with configuration.

        Args:
            config: Matrix configuration containing homeserver, user_id, access_token, etc.
        """
        self.config = config
        matrix_config = config.get("matrix", {})

        self.homeserver = matrix_config.get("homeserver", "https://matrix.org")
        self.user_id = matrix_config.get("user_id", "")
        self.access_token = matrix_config.get("access_token", "")
        self.device_id = matrix_config.get("device_id", "MATRIX_LLMAGENT")

        if not self.user_id or not self.access_token:
            raise ValueError("Matrix user_id and access_token are required in config")

        # E2EE configuration
        encryption_config = matrix_config.get("encryption", {})
        encryption_enabled = encryption_config.get("enabled", True)
        store_path = encryption_config.get("store_path", "./nio_store/")

        # Create store directory if encryption is enabled
        if encryption_enabled and store_path:
            os.makedirs(store_path, exist_ok=True)

        # Configure client for E2EE
        client_config = AsyncClientConfig(
            encryption_enabled=encryption_enabled,
            store_sync_tokens=True,
        )

        # Initialize AsyncClient with encryption support
        self.client = AsyncClient(
            self.homeserver,
            self.user_id,
            device_id=self.device_id,
            store_path=store_path if encryption_enabled else None,
            config=client_config,
        )
        self.client.access_token = self.access_token

        # For E2EE: load the crypto store (creates it if it doesn't exist)
        # Note: client.user_id must be set for load_store to work
        # AsyncClient sets client.user from constructor, but client.user_id is empty until login
        # We need to set it explicitly for load_store to work
        if encryption_enabled and self.user_id:
            self.client.user_id = self.user_id
            self.client.load_store()
            logger.info(f"Loaded crypto store from {store_path}")

        self._encryption_enabled = encryption_enabled
        logger.info(
            f"Matrix client initialized for {self.user_id} on {self.homeserver} "
            f"(E2EE: {'enabled' if encryption_enabled else 'disabled'})"
        )

    async def connect(self) -> None:
        """Connect to Matrix homeserver and verify credentials."""
        # Verify the access token works by getting our own user info
        response = await self.client.whoami()
        if hasattr(response, "user_id"):
            logger.info(f"Successfully connected to Matrix as {response.user_id}")
        else:
            logger.error(f"Failed to connect to Matrix: {response}")
            raise ConnectionError(f"Matrix connection failed: {response}")

        # For E2EE: do an initial sync to initialize the crypto store,
        # then upload keys if needed
        if self._encryption_enabled:
            logger.info("E2EE enabled, performing initial sync for crypto setup...")
            await self.client.sync(timeout=10000, full_state=True)

            if self.client.should_upload_keys:
                logger.info("Uploading encryption keys...")
                await self.client.keys_upload()

    async def sync(self, timeout: int = 30000) -> None:
        """Sync with Matrix server to receive events.

        Args:
            timeout: Sync timeout in milliseconds (default 30s)
        """
        response = await self.client.sync(timeout=timeout)

        # For E2EE: query and claim keys as needed
        if self._encryption_enabled:
            if self.client.should_query_keys:
                await self.client.keys_query()
            if self.client.should_claim_keys:
                await self.client.keys_claim(self.client.get_users_for_key_claiming())
        if hasattr(response, "rooms"):
            logger.debug(f"Sync completed, received {len(response.rooms.join)} room updates")
        return response

    async def send_message(self, room_id: str, message: str, msgtype: str = "m.text") -> None:
        """Send a text message to a room.

        Args:
            room_id: Matrix room ID (e.g., "!abc123:matrix.org")
            message: Message text to send
            msgtype: Message type (default: m.text)
        """
        content = {
            "msgtype": msgtype,
            "body": message,
        }
        response = await self.client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content=content,
            ignore_unverified_devices=True,  # Allow sending to unverified devices
        )
        if hasattr(response, "event_id"):
            logger.debug(f"Message sent to {room_id}: {response.event_id}")
        else:
            logger.error(f"Failed to send message to {room_id}: {response}")

    async def send_html_message(
        self, room_id: str, text: str, html: str, msgtype: str = "m.text"
    ) -> None:
        """Send a formatted HTML message to a room.

        Args:
            room_id: Matrix room ID
            text: Plain text fallback
            html: HTML formatted message
            msgtype: Message type (default: m.text)
        """
        content = {
            "msgtype": msgtype,
            "body": text,
            "format": "org.matrix.custom.html",
            "formatted_body": html,
        }
        response = await self.client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content=content,
            ignore_unverified_devices=True,  # Allow sending to unverified devices
        )
        if hasattr(response, "event_id"):
            logger.debug(f"HTML message sent to {room_id}: {response.event_id}")
        else:
            logger.error(f"Failed to send HTML message to {room_id}: {response}")

    async def get_display_name(self) -> str:
        """Get bot's display name.

        Returns:
            Bot's display name or user_id if not set
        """
        response = await self.client.get_displayname(self.user_id)
        if hasattr(response, "displayname") and response.displayname:
            return response.displayname
        return self.user_id

    async def join_room(self, room_id: str) -> None:
        """Join a Matrix room.

        Args:
            room_id: Room ID or alias to join
        """
        response = await self.client.join(room_id)
        if hasattr(response, "room_id"):
            logger.info(f"Joined room: {response.room_id}")
        else:
            logger.error(f"Failed to join room {room_id}: {response}")

    async def send_image(
        self,
        room_id: str,
        image_data: bytes,
        filename: str,
        mimetype: str = "image/png",
    ) -> None:
        """Upload and send an image to a room.

        Note: Element Web requires the body field to have a file extension
        to display images inline. The filename is used as body.
        Send any caption as a separate text message before calling this.

        Args:
            room_id: Matrix room ID
            image_data: Raw image bytes
            filename: Filename for the image (must have extension like .png)
            mimetype: MIME type (default: image/png)
        """
        # Upload image to Matrix content repository
        # nio.upload() expects a file-like object, not raw bytes
        data_stream = io.BytesIO(image_data)
        upload_response, _ = await self.client.upload(
            data_stream,
            content_type=mimetype,
            filename=filename,
            filesize=len(image_data),
        )

        if not hasattr(upload_response, "content_uri"):
            logger.error(f"Failed to upload image to Matrix: {upload_response}")
            return

        mxc_url = upload_response.content_uri
        logger.debug(f"Uploaded image to Matrix: {mxc_url}")

        # Try to get image dimensions for proper inline display
        width, height = None, None
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(image_data))
            width, height = img.size
        except Exception:
            pass  # Dimensions are optional

        # Send m.image message
        info: dict[str, Any] = {
            "mimetype": mimetype,
            "size": len(image_data),
        }
        if width and height:
            info["w"] = width
            info["h"] = height

        content: dict[str, Any] = {
            "msgtype": "m.image",
            "body": filename,
            "filename": filename,
            "url": mxc_url,
            "info": info,
        }

        response = await self.client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content=content,
            ignore_unverified_devices=True,  # Allow sending to unverified devices
        )

        if hasattr(response, "event_id"):
            logger.debug(f"Image sent to {room_id}: {response.event_id}")
        else:
            logger.error(f"Failed to send image to {room_id}: {response}")

    async def close(self) -> None:
        """Close the Matrix client connection."""
        await self.client.close()
        logger.info("Matrix client closed")

    def add_event_callback(self, callback, event_type) -> None:
        """Add a callback for specific Matrix events.

        Args:
            callback: Async function to call when event occurs
            event_type: Matrix event type to listen for (e.g., RoomMessageText)
        """
        self.client.add_event_callback(callback, event_type)
        logger.debug(f"Added event callback for {event_type}")

    def trust_devices_for_user(self, user_id: str) -> None:
        """Trust all devices for a given user (auto-trust mode).

        This is needed for E2EE - we must trust devices before we can
        decrypt their messages.

        Args:
            user_id: Matrix user ID to trust devices for
        """
        if not self._encryption_enabled:
            return

        # Check if olm is loaded
        if self.client.olm is None:
            logger.debug("Olm not loaded, skipping device trust")
            return

        if user_id not in self.client.device_store.users:
            logger.debug(f"No devices found for {user_id}")
            return

        for device_id, device in self.client.device_store[user_id].items():
            # Skip our own device
            if user_id == self.client.user_id and device_id == self.client.device_id:
                continue

            if not self.client.olm.is_device_verified(device):
                self.client.verify_device(device)
                logger.info(f"Trusted device {device_id} for user {user_id}")
