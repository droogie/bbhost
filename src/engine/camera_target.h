#pragma once

// Registered before the game starts; queried without retaining actor pointers.
namespace camera_target {
using Callback = void* (*)(void*, void*);
inline Callback callback = nullptr;
inline void* context = nullptr;

inline bool register_callback(Callback fn, void* user) {
    if (!fn || callback) return false;
    context = user;
    callback = fn;
    return true;
}

inline void* get(void* character) {
    return character && callback ? callback(character, context) : nullptr;
}
}  // namespace camera_target
