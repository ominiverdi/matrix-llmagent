#!/usr/bin/env python3
"""Get Matrix access token by logging in."""

import asyncio
import sys
from nio import AsyncClient, LoginResponse


async def get_access_token(homeserver, user_id, password):
    """Login and get access token."""
    print("Matrix Login - Get Access Token")
    print("=" * 50)

    # Create client
    client = AsyncClient(homeserver, user_id)

    try:
        print("\nLogging in...")
        response = await client.login(password, device_name="matrix-llmagent-setup")

        if isinstance(response, LoginResponse):
            print("\n✅ Login successful!")
            print("=" * 50)
            print(f"User ID:      {response.user_id}")
            print(f"Device ID:    {response.device_id}")
            print(f"Access Token: {response.access_token}")
            print("=" * 50)
            print("\n⚠️  IMPORTANT: Save these credentials securely!")
            print("Add them to your config.json file.")

            return response.access_token, response.device_id
        else:
            print(f"\n❌ Login failed: {response}")
            return None, None

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None, None
    finally:
        await client.close()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python get_token.py <homeserver> <user_id> <password>")
        print("Example: python get_token.py https://matrix.org @bot:matrix.org mypassword")
        sys.exit(1)

    homeserver = sys.argv[1]
    user_id = sys.argv[2]
    password = sys.argv[3]

    asyncio.run(get_access_token(homeserver, user_id, password))
