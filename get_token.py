#!/usr/bin/env python3
"""Get Matrix access token by logging in with a specific device_id."""

import asyncio
import sys

from nio import AsyncClient, LoginResponse


async def get_access_token(homeserver, user_id, password, device_id=None):
    """Login and get access token."""
    print("Matrix Login - Get Access Token")
    print("=" * 50)

    # Create client with optional device_id
    client = AsyncClient(homeserver, user_id, device_id=device_id)

    try:
        print(f"\nLogging in as {user_id}...")
        if device_id:
            print(f"Requesting device_id: {device_id}")

        # nio's login() doesn't accept device_id directly, it uses the client's device_id
        response = await client.login(password, device_name=device_id or "matrix-llmagent")

        if isinstance(response, LoginResponse):
            print("\nLogin successful!")
            print("=" * 50)
            print(f"User ID:      {response.user_id}")
            print(f"Device ID:    {response.device_id}")
            print(f"Access Token: {response.access_token}")
            print("=" * 50)
            print("\nIMPORTANT: Save these credentials securely!")
            print("Add them to your config.json file.")

            return response.access_token, response.device_id
        else:
            print(f"\nLogin failed: {response}")
            return None, None

    except Exception as e:
        print(f"\nError: {e}")
        return None, None
    finally:
        await client.close()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python get_token.py <homeserver> <user_id> <password> [device_id]")
        print(
            "Example: python get_token.py https://matrix.org @bot:matrix.org mypassword MYBOT_DEVICE"
        )
        sys.exit(1)

    homeserver = sys.argv[1]
    user_id = sys.argv[2]
    password = sys.argv[3]
    device_id = sys.argv[4] if len(sys.argv) > 4 else None

    asyncio.run(get_access_token(homeserver, user_id, password, device_id))
