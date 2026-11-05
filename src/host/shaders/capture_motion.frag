#version 450
// The captured meshes' motion (capture_motion.vert): NDC now minus NDC then in
// red and green, the fragment's view distance (clip w) in blue, alpha 0.5
// marking it - what camera_motion.comp reads, keeping it only where this
// fragment is the surface the depth holds. The nearest fragment wins (the
// pass's own depth test).
//
// Compiled offline (tools/build_shaders.sh) into capture_motion_frag.spv.h.
layout(location = 0) in vec4 now_clip;
layout(location = 1) in vec4 then_clip;
layout(location = 0) out vec4 o;

void main() {
    if (now_clip.w <= 0.05 || then_clip.w <= 0.05) discard;
    const vec2 d = now_clip.xy / now_clip.w - then_clip.xy / then_clip.w;
    if (any(isnan(d)) || any(isinf(d))) discard;
    o = vec4(d, now_clip.w, 0.5);
}
