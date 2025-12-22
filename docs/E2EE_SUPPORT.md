# End-to-End Encryption (E2EE) Support

## Current Status: Partial Implementation

E2EE support has been partially implemented but **does not reliably work** due to a fundamental Matrix protocol issue: clients like Element do not always share Megolm session keys with new/unverified devices.

### What's Implemented

- Crypto store creation and persistence (`./nio_store/`)
- Device key generation and upload
- Automatic key query/claim after sync
- `ignore_unverified_devices=True` for sending messages
- Graceful degradation (error message when decryption fails)
- `matrix-invites` CLI tool for managing encrypted room invitations

### What Doesn't Work

**Decryption of incoming messages fails** because:

1. When a user sends an encrypted message, their client (e.g., Element) creates a Megolm session
2. Element shares the Megolm session keys only with devices it knows about at that moment
3. Even though the bot's device keys are uploaded, Element often doesn't include the bot's device in the key sharing
4. The bot receives `MegolmEvent` but cannot decrypt it ("no session found")

This is a known Matrix ecosystem issue related to device list synchronization and key sharing timing.

## Workarounds

### Option 1: Use Unencrypted Rooms (Recommended)

To chat privately with the bot without encryption issues:

1. **Create a new room** (not "Start Direct Message")
   - In Element: Click "+" > "New Room"
   - Name it something like "Bot Chat"

2. **Disable encryption before sending any messages**
   - Room Settings > Security & Privacy > Turn OFF "Encrypted"
   - Note: Once encryption is enabled, it cannot be disabled

3. **Invite the bot** to the room

4. **Start chatting** - the bot will respond normally

### Option 2: Use the Invite Manager (Experimental)

The `matrix-invites` tool can create encrypted DMs where the bot initiates:

```bash
uv run matrix-invites
> dm @username:matrix.org
```

This creates an encrypted room and sends an initial greeting, attempting to establish the Megolm session first. However, subsequent messages from the user may still fail to decrypt due to Element creating new sessions.

### Option 3: /discardsession (Unreliable)

After the bot joins an encrypted room, the user can try:

1. Type `/discardsession` in the chat
2. Send a new message

This forces Element to create a new Megolm session, but it may still not include the bot's device.

## Technical Details

### Why E2EE is Complex for Bots

Matrix E2EE uses Olm/Megolm protocols:

| Component | Description |
|-----------|-------------|
| **Olm** | 1:1 encrypted channel between devices for key exchange |
| **Megolm** | Group encryption for room messages |
| **Device keys** | Each device has identity keys uploaded to homeserver |
| **Session keys** | Megolm session keys shared via Olm to authorized devices |

The problem: **Key sharing is sender-initiated**. When Alice sends a message, her client decides which devices get the Megolm session keys. If Alice's client doesn't have the bot's device in its device list (or doesn't trust it), the bot never gets the keys.

### Implementation Details

**Dependencies:**
```bash
# System
apt-get install libolm-dev

# Python
pip install "matrix-nio[e2e]"
```

**Key files:**
- `matrix_llmagent/matrix_client.py` - E2EE client configuration, `load_store()`, key upload
- `matrix_llmagent/matrix_monitor.py` - `MegolmEvent` handling
- `matrix_llmagent/invite_manager.py` - CLI for room management
- `./nio_store/` - SQLite crypto store (device keys, sessions)

**Config options:**
```json
{
  "matrix": {
    "encryption": {
      "enabled": true,
      "store_path": "./nio_store/"
    }
  }
}
```

### Known Limitations

1. **No cross-signing support** - matrix-nio cannot verify via master signing key
2. **No SSSS** - No server-side key backup integration
3. **No key request handling** - Cannot request keys for missed messages
4. **Device list sync issues** - Other clients may not see bot's device
5. **Session key sharing** - Depends entirely on sender's client behavior

## Future Options

### Pantalaimon

[Pantalaimon](https://github.com/matrix-org/pantalaimon) is an E2EE-aware proxy that sits between a client and homeserver. It handles all encryption/decryption, presenting unencrypted messages to the client. This could be a more reliable solution but adds deployment complexity.

### Dehydrated Devices

A future Matrix feature that would allow "drying" a device so it can receive keys while offline. Not yet widely implemented.

### Wait for Matrix Improvements

The Matrix protocol is actively being improved. Future versions may have better key sharing mechanisms.

## References

- [Fix Decryption Error Guide](https://joinmatrix.org/guide/fix-decryption-error/)
- [Unable to Decrypt Explained](https://blog.neko.dev/posts/unable-to-decrypt-matrix.html)
- [matrix-nio E2EE documentation](https://matrix-nio.readthedocs.io/en/latest/examples.html)
- [Pantalaimon](https://github.com/matrix-org/pantalaimon)
- [Matrix E2EE Spec](https://matrix.org/docs/matrix-concepts/end-to-end-encryption/)
