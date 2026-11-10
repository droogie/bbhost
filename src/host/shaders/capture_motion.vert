#version 450
// The captured meshes' motion (host/upscale_pass.cpp, BBNextGenSuite's method
// done natively): each corner of a skinned G-buffer mesh, its clip position
// this frame and last as its own vertex shader exported them (the translator's
// capture_position), drawn through the G-buffer's viewport - the jitter
// included - so it covers what the mesh covered.
//
// Compiled offline (tools/build_shaders.sh) into capture_motion_vert.spv.h.
layout(set = 0, binding = 0) readonly buffer Captured {
    vec4 pos[];  // every slot's corners, 16 bytes each
};
// The mesh's place in pos this frame and last; its own indices (each less its
// lowest, kept once per mesh in the capture's index buffer) are the vertex index.
layout(push_constant) uniform Bases {
    uint now_base;
    uint then_base;
};

layout(location = 0) out vec4 now_clip;
layout(location = 1) out vec4 then_clip;

void main() {
    const uint v = uint(gl_VertexIndex);
    now_clip = pos[now_base + v];
    then_clip = pos[then_base + v];
    gl_Position = now_clip;
}
