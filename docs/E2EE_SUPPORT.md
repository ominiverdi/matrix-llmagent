# End-to-End Encryption (E2EE) Support

## Current Status: Full Implementation with Cross-Signing

E2EE support is implemented with cross-signing and emoji verification. The bot is cross-signed by its owner, which enables trust via the cross-signing chain.

### What's Implemented

- Crypto store creation and persistence (`./nio_store/`)
- Device key generation and upload
- Automatic key query/claim after sync
- **Cross-signing support** - bot device is signed by owner's cross-signing keys
- **Device fingerprint display on startup** for manual verification
- **Emoji verification support** - bot automatically accepts and processes verification requests
- **`!verify` command** - users can request device info for manual verification
- `ignore_unverified_devices=True` for sending messages
- Graceful degradation (error message when decryption fails)
- `matrix-invites` CLI tool for managing encrypted room invitations

## How E2EE Works with the Bot

### The Trust Model

Matrix E2EE requires devices to be verified before encrypted messages work reliably. There are two ways to establish trust:

1. **Cross-signing chain** (recommended): Verify the bot's owner, and all their devices (including the bot) are automatically trusted
2. **Direct device verification**: Manually verify the bot's specific device via emoji or fingerprint

### Why "Unable to decrypt" Happens

When a user sends an encrypted message:
1. Their client (Element) creates Megolm session keys
2. These keys are shared only with **verified/trusted** devices
3. If the bot's device isn't trusted, it doesn't receive the keys
4. The bot can receive the message but cannot decrypt it

The same applies in reverse - if the user hasn't verified the bot, their client may not accept Megolm keys from the bot, showing "Unable to decrypt" for the bot's responses.

## Verifying the Bot

### Option 1: Cross-Signing Chain (Easiest)

If you verify the bot's owner (@ominiverdi:matrix.org), the bot is automatically trusted:

1. In Element, start a DM with @ominiverdi:matrix.org
2. Click their name > "Verify"
3. Complete emoji verification
4. The bot's device is now trusted via cross-signing

### Option 2: Use the !verify Command

The bot has a built-in command to help with verification:

1. Send `!verify` to the bot
2. You'll receive the bot's Device ID and fingerprint
3. Follow the instructions to manually verify

### Option 3: Manual Device Verification in Element

1. Click the bot's name in a chat
2. Click "View devices" or similar
3. Find the bot's device (e.g., `ksHi8AOCN8`)
4. Click "Manually verify by text"
5. Compare the fingerprint with what the bot shows

Note: Element Web has made this option harder to find in recent versions, preferring cross-signing.

### Option 4: Emoji Verification

1. In Element, find the bot's device and click "Verify"
2. Choose "Verify by emoji"
3. The bot automatically accepts and confirms
4. Check the bot's logs to see the emoji sequence
5. Confirm in Element if they match

## Alternative: Unencrypted Rooms

If verification is problematic, use unencrypted rooms:

1. **Create a new room** (not "Start Direct Message")
   - In Element: Click "+" > "New Room"
   - Name it something like "Bot Chat"

2. **Disable encryption before sending any messages**
   - Room Settings > Security & Privacy > Turn OFF "Encrypted"
   - Note: Once encryption is enabled, it cannot be disabled

3. **Invite the bot** to the room

4. **Start chatting** - the bot will respond normally

## Current Bot Configuration

```
User: @llm-assitant:matrix.org
Device ID: ksHi8AOCN8
Fingerprint: DUTJ dHYR hh9G wNYZ RTpF +vBj bxS4 u07+ wdu6 EX6e bqU
Cross-signed by: @ominiverdi:matrix.org
```

## Technical Details

### Cross-Signing Setup

The bot's cross-signing was set up via:

1. Reset identity at https://account.matrix.org/account/?action=org.matrix.cross_signing_reset
2. Set up Secure Backup in Element Web
3. Used `mx` CLI tool to download cross-signing keys and verify the bot's device

Recovery key is stored securely for key recovery if needed.

### Why E2EE is Complex for Bots

Matrix E2EE uses Olm/Megolm protocols:

| Component | Description |
|-----------|-------------|
| **Olm** | 1:1 encrypted channel between devices for key exchange |
| **Megolm** | Group encryption for room messages |
| **Device keys** | Each device has identity keys uploaded to homeserver |
| **Session keys** | Megolm session keys shared via Olm to authorized devices |
| **Cross-signing** | User-level trust that extends to all their devices |
| **Device verification** | Direct trust establishment between specific devices |

The key insight: **Key sharing is sender-initiated and trust-based**. When you send a message, your client decides which devices get the Megolm session keys based on verification status.

### Implementation Details

**Dependencies:**
```bash
# System
apt-get install libolm-dev

# Python
pip install "matrix-nio[e2e]"
```

**Key files:**
- `matrix_llmagent/matrix_client.py` - E2EE client, fingerprint display, verification handler
- `matrix_llmagent/matrix_monitor.py` - MegolmEvent handling, `!verify` command, verification callback
- `matrix_llmagent/invite_manager.py` - CLI for room management
- `./nio_store/` - SQLite crypto store (device keys, sessions)

**Config options:**
```json
{
  "matrix": {
    "device_id": "ksHi8AOCN8",
    "encryption": {
      "enabled": true,
      "store_path": "./nio_store/"
    }
  }
}
```

**Important:** Keep the same `device_id` and `access_token` across restarts. Changing them creates a new device identity, invalidating existing verifications and cross-signing.

### The Verification Flow

```
User                          Bot                         Element
  |                            |                            |
  |  Start verification -----> |                            |
  |                            | <-- Accept verification    |
  |                            | --> Share key              |
  | <-- Show emojis            |                            |
  |                            | --> Auto-confirm emojis    |
  | Confirm match -----------> |                            |
  |                            | <-- Verification complete  |
  |                            |                            |
  | Send encrypted message --> |                            |
  |                            | (has keys, can decrypt!)   |
  | <-- Encrypted response --- |                            |
```

### Troubleshooting

**"Unable to decrypt" from the bot:**
- You need to verify the bot's device (see options above)
- Try the `!verify` command to get verification instructions
- Alternatively, verify @ominiverdi:matrix.org to trust via cross-signing

**"Unable to decrypt" when sending TO the bot:**
- The bot auto-trusts sender devices, so this is less common
- Try `/discardsession` in the chat to force a new Megolm session
- The bot will send an error message if it can't decrypt

**Verification request not working:**
- Ensure the bot is running and connected
- Check the bot's logs for KeyVerification events
- Try manual verification instead of emoji

**Bot shows as "unverified" after restart:**
- The `device_id` must be the same across restarts
- The `./nio_store/` directory must persist
- If you lose the store, you need to re-verify and re-establish cross-signing

## Tools

### mx CLI

The `mx` CLI tool (https://codeberg.org/andybalholm/mx) is useful for:
- Downloading cross-signing keys: `mx recovery --recovery-key "..." download`
- Verifying devices: `mx identity verify-device <device_id> <fingerprint>`

Build from source (use commit `22c6f3c` or later):
```bash
git clone https://codeberg.org/andybalholm/mx
cd mx
cargo build --release
```

### matrix-invites

Built-in CLI for managing room invitations:
```bash
uv run matrix-invites --help
```

## References

- [matrix-nio E2EE examples](https://matrix-nio.readthedocs.io/en/latest/examples.html)
- [Matrix E2EE implementation guide](https://matrix.org/docs/matrix-concepts/end-to-end-encryption/)
- [Fix Decryption Error Guide](https://joinmatrix.org/guide/fix-decryption-error/)
- [The UISI Bug](https://github.com/element-hq/element-web/issues/2996) - detailed explanation of decryption issues
