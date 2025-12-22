#!/usr/bin/env python3
"""CLI tool to manage Matrix room invitations for the bot.

This is needed for E2EE rooms: the bot must accept invites itself
(not via another client like Fractal) to receive encryption keys.

Usage:
    uv run python -m matrix_llmagent.invite_manager [--config CONFIG]

Commands:
    list    - Show pending invitations
    accept  - Accept an invitation by room ID or number
    reject  - Reject an invitation by room ID or number
    quit    - Exit the manager
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from nio import AsyncClient, AsyncClientConfig, JoinError, RoomLeaveError

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


async def get_pending_invites(client: AsyncClient) -> list[dict]:
    """Get list of pending room invitations."""
    invites = []
    for room_id, room in client.invited_rooms.items():
        invite_info = {
            "room_id": room_id,
            "name": room.display_name or room_id,
            "inviter": None,
        }
        # Try to find who invited us
        for event in room.invite_state:
            if hasattr(event, "sender"):
                invite_info["inviter"] = event.sender
                break
        invites.append(invite_info)
    return invites


async def accept_invite(client: AsyncClient, room_id: str) -> bool:
    """Accept a room invitation."""
    response = await client.join(room_id)
    if isinstance(response, JoinError):
        logger.error(f"Failed to join {room_id}: {response.message}")
        return False
    logger.info(f"Joined room: {room_id}")
    return True


async def reject_invite(client: AsyncClient, room_id: str) -> bool:
    """Reject a room invitation."""
    response = await client.room_leave(room_id)
    if isinstance(response, RoomLeaveError):
        logger.error(f"Failed to reject {room_id}: {response.message}")
        return False
    logger.info(f"Rejected invite for room: {room_id}")
    return True


def print_invites(invites: list[dict]) -> None:
    """Print pending invitations."""
    if not invites:
        print("\nNo pending invitations.\n")
        return

    print(f"\nPending invitations ({len(invites)}):")
    print("-" * 60)
    for i, inv in enumerate(invites, 1):
        inviter = inv["inviter"] or "unknown"
        print(f"  [{i}] {inv['name']}")
        print(f"      Room ID: {inv['room_id']}")
        print(f"      Invited by: {inviter}")
        print()


async def interactive_loop(client: AsyncClient) -> None:
    """Run interactive command loop."""
    print("\nMatrix Invite Manager")
    print("=" * 40)
    print("Commands: list, accept <n>, reject <n>, quit")
    print()

    while True:
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not cmd:
            continue

        parts = cmd.split()
        action = parts[0]

        if action in ("quit", "exit", "q"):
            print("Exiting...")
            break

        elif action in ("list", "ls", "l"):
            # Refresh sync to get latest invites
            await client.sync(timeout=5000)
            invites = await get_pending_invites(client)
            print_invites(invites)

        elif action in ("accept", "a", "join", "j"):
            if len(parts) < 2:
                print("Usage: accept <number or room_id>")
                continue

            await client.sync(timeout=5000)
            invites = await get_pending_invites(client)

            target = parts[1]
            room_id = None

            # Try as number first
            try:
                idx = int(target)
                if 1 <= idx <= len(invites):
                    room_id = invites[idx - 1]["room_id"]
                else:
                    print(f"Invalid number. Use 1-{len(invites)}")
                    continue
            except ValueError:
                # Treat as room_id
                room_id = target

            if room_id:
                await accept_invite(client, room_id)

        elif action in ("reject", "r", "decline", "d"):
            if len(parts) < 2:
                print("Usage: reject <number or room_id>")
                continue

            await client.sync(timeout=5000)
            invites = await get_pending_invites(client)

            target = parts[1]
            room_id = None

            # Try as number first
            try:
                idx = int(target)
                if 1 <= idx <= len(invites):
                    room_id = invites[idx - 1]["room_id"]
                else:
                    print(f"Invalid number. Use 1-{len(invites)}")
                    continue
            except ValueError:
                # Treat as room_id
                room_id = target

            if room_id:
                await reject_invite(client, room_id)

        else:
            print("Unknown command. Use: list, accept <n>, reject <n>, quit")


async def main(config_path: str) -> None:
    """Main entry point."""
    # Load config
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_file) as f:
        config = json.load(f)

    matrix_config = config.get("matrix", {})
    homeserver = matrix_config.get("homeserver", "https://matrix.org")
    user_id = matrix_config.get("user_id", "")
    access_token = matrix_config.get("access_token", "")
    device_id = matrix_config.get("device_id", "MATRIX_LLMAGENT")

    if not user_id or not access_token:
        logger.error("Matrix user_id and access_token required in config")
        sys.exit(1)

    # E2EE configuration
    encryption_config = matrix_config.get("encryption", {})
    encryption_enabled = encryption_config.get("enabled", True)
    store_path = encryption_config.get("store_path", "./nio_store/")

    if encryption_enabled and store_path:
        Path(store_path).mkdir(parents=True, exist_ok=True)

    client_config = AsyncClientConfig(
        encryption_enabled=encryption_enabled,
        store_sync_tokens=True,
    )

    client = AsyncClient(
        homeserver,
        user_id,
        device_id=device_id,
        store_path=store_path if encryption_enabled else None,
        config=client_config,
    )
    client.access_token = access_token

    try:
        # Initial sync
        print(f"Connecting as {user_id}...")
        await client.sync(timeout=10000, full_state=True)

        # Upload keys if needed
        if encryption_enabled and client.should_upload_keys:
            print("Uploading encryption keys...")
            await client.keys_upload()

        # Show initial invites
        invites = await get_pending_invites(client)
        print_invites(invites)

        # Interactive loop
        await interactive_loop(client)

    finally:
        await client.close()


def run() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Manage Matrix room invitations for the bot")
    parser.add_argument(
        "--config",
        "-c",
        default="config.json",
        help="Path to config file (default: config.json)",
    )
    args = parser.parse_args()

    asyncio.run(main(args.config))


if __name__ == "__main__":
    run()
