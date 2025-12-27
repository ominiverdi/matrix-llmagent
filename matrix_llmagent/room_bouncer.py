#!/usr/bin/env python3
"""Room Bouncer - Manage Matrix room membership for the bot.

Usage:
    uv run python -m matrix_llmagent.room_bouncer [--config CONFIG]

Commands:
    list    - Show all joined rooms
    leave   - Leave a room by number or room ID
    quit    - Exit the bouncer
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from nio import AsyncClient, AsyncClientConfig, RoomLeaveError

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy nio logging
logging.getLogger("nio").setLevel(logging.WARNING)


async def get_joined_rooms(client: AsyncClient) -> list[dict]:
    """Get list of joined rooms with details."""
    rooms = []
    for room_id, room in client.rooms.items():
        room_info = {
            "room_id": room_id,
            "name": room.display_name or room_id,
            "member_count": room.member_count,
            "encrypted": room.encrypted,
        }
        rooms.append(room_info)
    # Sort by name
    rooms.sort(key=lambda r: r["name"].lower())
    return rooms


async def leave_room(client: AsyncClient, room_id: str) -> bool:
    """Leave a room."""
    response = await client.room_leave(room_id)
    if isinstance(response, RoomLeaveError):
        logger.error(f"Failed to leave {room_id}: {response.message}")
        return False
    logger.info(f"Left room: {room_id}")
    return True


def print_rooms(rooms: list[dict]) -> None:
    """Print joined rooms."""
    if not rooms:
        print("\nNo joined rooms.\n")
        return

    print(f"\nJoined rooms ({len(rooms)}):")
    print("-" * 70)
    for i, room in enumerate(rooms, 1):
        encrypted = "E2EE" if room["encrypted"] else "plain"
        print(f"  [{i:2}] {room['name']}")
        print(f"       {room['room_id']}")
        print(f"       {room['member_count']} members, {encrypted}")
        print()


async def interactive_loop(client: AsyncClient) -> None:
    """Run interactive command loop."""
    print("\nRoom Bouncer")
    print("=" * 40)
    print("Commands: list, leave <n>, quit")
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
            await client.sync(timeout=5000)
            rooms = await get_joined_rooms(client)
            print_rooms(rooms)

        elif action in ("leave", "kick", "bounce", "bye"):
            if len(parts) < 2:
                print("Usage: leave <number or room_id>")
                continue

            await client.sync(timeout=5000)
            rooms = await get_joined_rooms(client)

            target = parts[1]
            room_id = None
            room_name = None

            # Try as number first
            try:
                idx = int(target)
                if 1 <= idx <= len(rooms):
                    room_id = rooms[idx - 1]["room_id"]
                    room_name = rooms[idx - 1]["name"]
                else:
                    print(f"Invalid number. Use 1-{len(rooms)}")
                    continue
            except ValueError:
                # Treat as room_id
                room_id = target
                for r in rooms:
                    if r["room_id"] == target:
                        room_name = r["name"]
                        break

            if room_id:
                confirm = input(f"Leave '{room_name or room_id}'? [y/N] ").strip().lower()
                if confirm in ("y", "yes"):
                    await leave_room(client, room_id)
                else:
                    print("Cancelled.")

        else:
            print("Commands: list, leave <n>, quit")


async def main(config_path: str) -> None:
    """Main entry point."""
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

    if encryption_enabled and user_id:
        client.user_id = user_id
        client.load_store()

    try:
        print(f"Connecting as {user_id}...")
        await client.sync(timeout=10000, full_state=True)

        # Show initial room list
        rooms = await get_joined_rooms(client)
        print_rooms(rooms)

        # Interactive loop
        await interactive_loop(client)

    finally:
        await client.close()


def run() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Manage Matrix room membership for the bot")
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
