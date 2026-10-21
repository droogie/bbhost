#include "discord_rpc.h"

// BBHost's IPC transport implementation; the public C declarations are from
// Discord's MIT-licensed discord-rpc header, distributed with its license.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

namespace {

bool g_initialized = false;
bool g_connected = false;
std::string g_appId;
std::uint64_t g_nonce = 0;
std::chrono::steady_clock::time_point g_next_connect_attempt{};

#ifdef _WIN32
HANDLE g_discordPipe = INVALID_HANDLE_VALUE;
#else
int g_discordSocket = -1;
#endif

std::string json_escape(const char* value) {
    std::string escaped;
    if (!value) return escaped;
    for (const unsigned char c : std::string(value)) {
        switch (c) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (c < 0x20) {
                    constexpr char hex[] = "0123456789abcdef";
                    escaped += "\\u00";
                    escaped += hex[c >> 4];
                    escaped += hex[c & 0x0f];
                } else {
                    escaped += static_cast<char>(c);
                }
        }
    }
    return escaped;
}

#ifdef _WIN32
bool write_all(const void* data, std::size_t size) {
    const char* cursor = static_cast<const char*>(data);
    while (size) {
        DWORD written = 0;
        const DWORD chunk = static_cast<DWORD>(size > 0x7fffffffu ? 0x7fffffffu : size);
        if (!WriteFile(g_discordPipe, cursor, chunk, &written, nullptr) || written == 0) return false;
        cursor += written;
        size -= written;
    }
    return true;
}

bool read_all(void* data, std::size_t size) {
    char* cursor = static_cast<char*>(data);
    while (size) {
        DWORD read = 0;
        const DWORD chunk = static_cast<DWORD>(size > 0x7fffffffu ? 0x7fffffffu : size);
        if (!ReadFile(g_discordPipe, cursor, chunk, &read, nullptr) || read == 0) return false;
        cursor += read;
        size -= read;
    }
    return true;
}
#endif

bool send_frame(std::uint32_t opcode, const std::string& payload) {
    const std::uint32_t header[2] = {opcode, static_cast<std::uint32_t>(payload.size())};
#ifdef _WIN32
    if (g_discordPipe == INVALID_HANDLE_VALUE) return false;
    return write_all(header, sizeof(header)) && write_all(payload.data(), payload.size());
#else
    if (g_discordSocket < 0) return false;
    auto send_bytes = [](const void* data, std::size_t size) {
        const char* cursor = static_cast<const char*>(data);
        while (size) {
            const ssize_t sent = ::send(g_discordSocket, cursor, size, MSG_NOSIGNAL);
            if (sent <= 0) return false;
            cursor += sent;
            size -= static_cast<std::size_t>(sent);
        }
        return true;
    };
    return send_bytes(header, sizeof(header)) && send_bytes(payload.data(), payload.size());
#endif
}

#ifdef _WIN32
bool receive_frame() {
    // Discord replies to the handshake with a framed READY event. Consume the
    // entire response before sending activity commands on the byte-mode pipe.
    std::uint32_t header[2]{};
    if (!read_all(header, sizeof(header)) || header[1] > 1024u * 1024u) return false;
    std::string response(header[1], '\0');
    return response.empty() || read_all(&response[0], response.size());
}

bool connect_windows_pipe() {
    for (int i = 0; i < 10; ++i) {
        const std::string path = "\\\\.\\pipe\\discord-ipc-" + std::to_string(i);
        g_discordPipe = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                                    OPEN_EXISTING, 0, nullptr);
        if (g_discordPipe != INVALID_HANDLE_VALUE) {
            std::printf("[bbhost] Discord RPC connected to pipe: %s\n", path.c_str());
            return true;
        }
    }
    return false;
}
#endif

bool connect_discord() {
#ifdef _WIN32
    if (!connect_windows_pipe()) return false;
    const std::string hello = "{\"v\":1,\"client_id\":\"" + json_escape(g_appId.c_str()) + "\"}";
    if (!send_frame(0, hello) || !receive_frame()) {
        std::printf("[bbhost] Discord IPC handshake failed.\n");
        CloseHandle(g_discordPipe);
        g_discordPipe = INVALID_HANDLE_VALUE;
        return false;
    }
    return true;
#else
    const char* runtime = std::getenv("XDG_RUNTIME_DIR");
    const char* home = std::getenv("HOME");
    for (int i = 0; i < 10; ++i) {
        std::string path = runtime && runtime[0] ? runtime : home ? std::string(home) + "/.config" : "/tmp";
        path += "/discord-ipc-" + std::to_string(i);
        if (path.size() >= sizeof(sockaddr_un::sun_path)) continue;
        g_discordSocket = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (g_discordSocket < 0) return false;
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
        if (::connect(g_discordSocket, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) {
            const std::string hello = "{\"v\":1,\"client_id\":\"" + json_escape(g_appId.c_str()) + "\"}";
            const std::uint32_t header[2] = {0, static_cast<std::uint32_t>(hello.size())};
            const bool sent = ::send(g_discordSocket, header, sizeof(header), MSG_NOSIGNAL) == sizeof(header) &&
                              ::send(g_discordSocket, hello.data(), hello.size(), MSG_NOSIGNAL) == static_cast<ssize_t>(hello.size());
            if (!sent) {
                ::close(g_discordSocket);
                g_discordSocket = -1;
            }
            return sent;
        }
        ::close(g_discordSocket);
        g_discordSocket = -1;
    }
    return false;
#endif
}

bool reconnect_if_due() {
    if (g_connected) return true;
    const auto now = std::chrono::steady_clock::now();
    if (now < g_next_connect_attempt) return false;
    g_next_connect_attempt = now + std::chrono::seconds(5);
    g_connected = connect_discord();
    if (g_connected) std::printf("[bbhost] Discord IPC reconnected.\n");
    return g_connected;
}

std::string make_activity_payload(const DiscordRichPresence* presence) {
    std::ostringstream json;
    json << "{\"cmd\":\"SET_ACTIVITY\",\"args\":{\"pid\":";
#ifdef _WIN32
    json << static_cast<unsigned long>(GetCurrentProcessId());
#else
    json << static_cast<long>(getpid());
#endif
    json << ",\"activity\":";
    if (!presence) {
        json << "null";
    } else {
        json << '{';
        bool comma = false;
        auto field = [&](const char* key, const char* value) {
            if (!value || !value[0]) return;
            if (comma) json << ',';
            json << '\"' << key << "\":\"" << json_escape(value) << '\"';
            comma = true;
        };
        field("details", presence->details);
        field("state", presence->state);
        if (presence->startTimestamp > 0) {
            if (comma) json << ',';
            json << "\"timestamps\":{\"start\":" << presence->startTimestamp << '}';
            comma = true;
        }
        if (presence->largeImageKey || presence->largeImageText || presence->smallImageKey || presence->smallImageText) {
            if (comma) json << ',';
            json << "\"assets\":{";
            bool asset_comma = false;
            auto asset = [&](const char* key, const char* value) {
                if (!value || !value[0]) return;
                if (asset_comma) json << ',';
                json << '\"' << key << "\":\"" << json_escape(value) << '\"';
                asset_comma = true;
            };
            asset("large_image", presence->largeImageKey);
            asset("large_text", presence->largeImageText);
            asset("small_image", presence->smallImageKey);
            asset("small_text", presence->smallImageText);
            json << '}';
            comma = true;
        }
        json << '}';
    }
    json << "},\"nonce\":\"" << ++g_nonce << "\"}";
    return json.str();
}

}  // namespace

extern "C" {

void Discord_Initialize(const char* applicationId, DiscordEventHandlers* handlers, int autoRegister, const char* optionalSteamId) {
    (void)handlers;
    (void)autoRegister;
    (void)optionalSteamId;
    if (g_initialized) return;
    if (applicationId) g_appId = applicationId;
    std::printf("[bbhost] Discord_Initialize called with App ID: %s\n", g_appId.c_str());
    g_next_connect_attempt = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    g_connected = connect_discord();
    g_initialized = true;
    if (g_connected) {
        std::printf("[bbhost] Discord IPC handshake completed.\n");
    } else {
        std::printf("[bbhost] Warning: Discord IPC connection/handshake failed; activity cannot be published.\n");
    }
}

void Discord_Shutdown(void) {
    if (g_initialized && g_connected) Discord_ClearPresence();
    g_initialized = false;
    g_connected = false;
    std::printf("[bbhost] Discord_Shutdown called.\n");
#ifdef _WIN32
    if (g_discordPipe != INVALID_HANDLE_VALUE) {
        CloseHandle(g_discordPipe);
        g_discordPipe = INVALID_HANDLE_VALUE;
    }
#else
    if (g_discordSocket >= 0) {
        ::close(g_discordSocket);
        g_discordSocket = -1;
    }
#endif
}

void Discord_RunCallbacks(void) {
    // This lightweight client does not expose callback delivery.
}

void Discord_UpdatePresence(const DiscordRichPresence* presence) {
    if (!g_initialized || !presence) return;
    std::printf("[bbhost] Discord_UpdatePresence -> Details: '%s' | State: '%s'\n",
                presence->details ? presence->details : "None",
                presence->state ? presence->state : "None");
    if ((!g_connected && !reconnect_if_due()) || !send_frame(1, make_activity_payload(presence))) {
        std::printf("[bbhost] Discord SET_ACTIVITY IPC write failed.\n");
        g_connected = false;
#ifdef _WIN32
        if (g_discordPipe != INVALID_HANDLE_VALUE) {
            CloseHandle(g_discordPipe);
            g_discordPipe = INVALID_HANDLE_VALUE;
        }
#else
        if (g_discordSocket >= 0) {
            ::close(g_discordSocket);
            g_discordSocket = -1;
        }
#endif
    }
}

void Discord_ClearPresence(void) {
    if (!g_initialized || !g_connected) return;
    if (!send_frame(1, make_activity_payload(nullptr))) {
        std::printf("[bbhost] Discord clear-activity IPC write failed.\n");
        g_connected = false;
    }
}

void Discord_Respond(const char* userId, int reply) {
    (void)userId;
    (void)reply;
}

void Discord_UpdateHandlers(DiscordEventHandlers* handlers) {
    (void)handlers;
}

}  // extern "C"
