# End-to-End Encryption (E2EE) Support

## Current Status: Full Implementation with Device Verification

E2EE support is implemented with emoji verification. For E2EE to work reliably, users must **verify the bot's device** before sending encrypted messages.

### What's Implemented

- Crypto store creation and persistence (`./nio_store/`)
- Device key generation and upload
- Automatic key query/claim after sync
- **Device fingerprint display on startup** for manual verification
- **Emoji verification support** - bot automatically accepts and processes verification requests
- `ignore_unverified_devices=True` for sending messages
- Graceful degradation (error message when decryption fails)
- `matrix-invites` CLI tool for managing encrypted room invitations

## How to Use E2EE with the Bot

### Step 1: Note the Bot's Device Fingerprint

When the bot starts, it logs its device fingerprint:

```
Device ID: MATRIX_LLMAGENT
Device fingerprint (Ed25519): jD1m e6gr WGFw ZrW9 f4mJ NqEP uFL+ ARQv JENu Dkuz aiA
To verify this bot, check this fingerprint matches in Element...
```

### Step 2: Verify the Bot's Device (Required for E2EE)

In Element (or your Matrix client):

1. Go to **Settings > Security & Privacy > Cross-signing**
2. Find the bot's user (e.g., `@bot:matrix.org`)
3. Click on the bot's device (e.g., `MATRIX_LLMAGENT`)
4. Choose **"Verify by emoji"** or **"Manually verify by text"**

**For emoji verification:**
- Element will send a verification request to the bot
- The bot automatically accepts and confirms
- Check the console log to see the emoji sequence
- Confirm in Element if they match

**For manual verification:**
- Compare the fingerprint in Element with the one the bot logged at startup
- If they match, click "Verify"

### Step 3: Send Encrypted Messages

After verification, Element will share Megolm session keys with the bot, and encrypted messages will work.

## Why Verification is Required

Matrix E2EE is designed so that clients only share encryption keys with **verified** or **known** devices. Without verification:

1. Your client (Element) sees the bot's device as "unverified"
2. When you send a message, Element may not share the Megolm keys with unverified devices
3. The bot receives the encrypted message but cannot decrypt it

Verification tells Element: "I trust this device, share keys with it."

## Alternative: Unencrypted Rooms

If you don't want to deal with verification, use unencrypted rooms:

1. **Create a new room** (not "Start Direct Message")
   - In Element: Click "+" > "New Room"
   - Name it something like "Bot Chat"

2. **Disable encryption before sending any messages**
   - Room Settings > Security & Privacy > Turn OFF "Encrypted"
   - Note: Once encryption is enabled, it cannot be disabled

3. **Invite the bot** to the room

4. **Start chatting** - the bot will respond normally

## Technical Details

### Why E2EE is Complex for Bots

Matrix E2EE uses Olm/Megolm protocols:

| Component | Description |
|-----------|-------------|
| **Olm** | 1:1 encrypted channel between devices for key exchange |
| **Megolm** | Group encryption for room messages |
| **Device keys** | Each device has identity keys uploaded to homeserver |
| **Session keys** | Megolm session keys shared via Olm to authorized devices |
| **Device verification** | Trust establishment so clients share keys |

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
- `matrix_llmagent/matrix_monitor.py` - MegolmEvent handling, verification callback setup
- `matrix_llmagent/invite_manager.py` - CLI for room management
- `./nio_store/` - SQLite crypto store (device keys, sessions)

**Config options:**
```json
{
  "matrix": {
    "device_id": "MATRIX_LLMAGENT",
    "encryption": {
      "enabled": true,
      "store_path": "./nio_store/"
    }
  }
}
```

**Important:** Keep the same `device_id` across restarts. Changing it creates a new device identity, invalidating existing verifications.

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

**"Unable to decrypt" after verification:**
- Try `/discardsession` in the chat to force a new Megolm session
- Make sure the bot wasn't restarted with a different `device_id`
- Check that `./nio_store/` is persistent (not deleted between restarts)

**Verification request not working:**
- Ensure the bot is running and connected
- Check the bot's console for verification events
- Try manual verification instead of emoji

**Bot shows as "unverified" after restart:**
- The `device_id` must be the same across restarts
- The `./nio_store/` directory must persist
- If you lose the store, you need to re-verify

## References

- [matrix-nio E2EE examples](https://matrix-nio.readthedocs.io/en/latest/examples.html)
- [Matrix E2EE implementation guide](https://matrix.org/docs/matrix-concepts/end-to-end-encryption/)
- [Fix Decryption Error Guide](https://joinmatrix.org/guide/fix-decryption-error/)
- [The UISI Bug](https://github.com/element-hq/element-web/issues/2996) - detailed explanation of decryption issues
