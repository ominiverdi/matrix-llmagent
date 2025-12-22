# End-to-End Encryption (E2EE) Support

## Current Status

The bot does **not** currently support end-to-end encrypted rooms. When invited to an encrypted room (which is the default for Matrix direct messages), the bot will:

- Join the room successfully
- Receive encrypted messages (as `MegolmEvent`)
- Be unable to decrypt or respond to them

## Workaround: Create Unencrypted Rooms

To chat privately with the bot without encryption:

1. **Create a new room** (not "Start Direct Message")
   - In Element: Click "+" > "New Room"
   - Name it something like "Bot Chat"

2. **Disable encryption before sending any messages**
   - Room Settings > Security & Privacy > Turn OFF "Encrypted"
   - Note: Once encryption is enabled, it cannot be disabled

3. **Invite the bot** to the room

4. **Start chatting** - the bot will now respond normally

## Why E2EE is Complex for Bots

Matrix E2EE uses the Olm/Megolm cryptographic protocols, which require:

| Requirement | Description |
|-------------|-------------|
| **libolm library** | Native C library for cryptographic operations |
| **Device keys** | Each bot instance needs persistent identity keys |
| **Session storage** | SQLite database for Megolm session keys |
| **Device verification** | Trust establishment with other devices |
| **Key persistence** | Loss of keys = loss of message history |

## Implementation Plan

### Dependencies Required

**System packages:**
```bash
# Ubuntu/Debian
apt-get install libolm-dev

# macOS
brew install libolm

# Alpine (Docker)
apk add olm-dev
```

**Python (pyproject.toml change):**
```diff
-    "matrix-nio>=0.24.0",
+    "matrix-nio[e2e]>=0.24.0",
```

### Code Changes Required

#### 1. Client Configuration (matrix_client.py)

```python
from nio import AsyncClient, AsyncClientConfig

# E2EE configuration
client_config = AsyncClientConfig(
    encryption_enabled=True,
    store_sync_tokens=True,
)

# Client needs store_path for key persistence
client = AsyncClient(
    homeserver=self.homeserver,
    user=self.user_id,
    device_id=self.device_id,
    store_path="./nio_store/",  # New: encryption key storage
    config=client_config,
)
```

#### 2. Credential Persistence

To maintain device identity across restarts, credentials must be saved:

```python
# After login, save credentials
credentials = {
    "homeserver": homeserver,
    "user_id": response.user_id,
    "device_id": response.device_id,
    "access_token": response.access_token,
}
# Save to secure storage (e.g., encrypted JSON file)

# On restart, restore instead of fresh login
client.restore_login(
    user_id=creds["user_id"],
    device_id=creds["device_id"],
    access_token=creds["access_token"],
)
client.load_store()  # Load encryption keys
```

#### 3. Key Upload

After login/restore, upload device keys to homeserver:

```python
if client.should_upload_keys:
    await client.keys_upload()
```

#### 4. Device Trust Strategy

**Option A: Auto-trust all devices (simple, less secure)**
```python
# Trust all devices for a user
for device_id, device in client.device_store[user_id].items():
    if not client.olm.is_device_verified(device):
        client.verify_device(device)
```

**Option B: Interactive emoji verification (complex, secure)**
- Requires handling `KeyVerificationStart`, `KeyVerificationKey`, `KeyVerificationMac` events
- User must compare emoji on both devices
- More appropriate for human-to-human verification

**Recommendation for bots:** Option A with clear documentation that auto-trust is enabled.

#### 5. Graceful Degradation

Handle messages that cannot be decrypted:

```python
from nio import MegolmEvent

async def on_room_message(room, event):
    if isinstance(event, MegolmEvent):
        # Could not decrypt
        logger.warning(f"Could not decrypt message in {room.room_id}")
        await client.send_message(
            room.room_id,
            "I received an encrypted message but couldn't decrypt it. "
            "This may happen with messages sent before I joined."
        )
        return
    # Normal message handling...
```

### Configuration Options

New config section for E2EE:

```json
{
  "matrix": {
    "encryption": {
      "enabled": true,
      "store_path": "./nio_store/",
      "auto_trust_devices": true,
      "credentials_file": "./matrix_credentials.json"
    }
  }
}
```

### Storage Requirements

| File/Directory | Purpose | Size |
|----------------|---------|------|
| `./nio_store/` | Encryption keys, sessions | ~100-500KB |
| `matrix_credentials.json` | Device ID, access token | ~1KB |

**Important:** The `nio_store/` directory contains cryptographic keys. If lost:
- Bot gets a new device identity
- Old encrypted messages become unreadable
- Users may need to re-verify the bot

### Docker Considerations

For containerized deployments:

```dockerfile
# Add libolm
RUN apt-get update && apt-get install -y libolm-dev

# Mount persistent volume for encryption keys
VOLUME ["/app/nio_store"]
```

```yaml
# docker-compose.yml
volumes:
  - ./nio_store:/app/nio_store
  - ./matrix_credentials.json:/app/matrix_credentials.json
```

### Limitations

matrix-nio E2EE does not support:

- **Cross-signing** - Cannot verify via master signing key
- **Server-side key backup** - No SSSS (Secure Secret Storage and Sharing)
- **Key sharing requests** - Cannot request keys for old messages

### Effort Estimate

| Task | Time |
|------|------|
| Dependencies + basic setup | 1-2 hours |
| Credential persistence | 1-2 hours |
| Auto-trust implementation | 1 hour |
| Testing with encrypted room | 1-2 hours |
| Docker updates | 30 min |
| Documentation | 30 min |
| **Total (minimal E2EE)** | **4-8 hours** |

Add 4-8 hours more for interactive verification support.

## References

- [matrix-nio E2EE documentation](https://matrix-nio.readthedocs.io/en/latest/examples.html#e2e-encryption)
- [matrix-nio examples](https://github.com/matrix-nio/matrix-nio/tree/main/examples)
- [Olm/Megolm specification](https://matrix.org/docs/matrix-concepts/end-to-end-encryption/)
