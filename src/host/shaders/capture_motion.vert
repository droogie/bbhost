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
layout(set = 0, binding = 1) readonly buffer Corners {
    uvec2 corner[];  // this frame's place in pos, the last frame's
};

layout(location = 0) out vec4 now_clip;
layout(location = 1) out vec4 then_clip;

void main() {
    const uvec2 c = corner[gl_VertexIndex];
    now_clip = pos[c.x];
    then_clip = pos[c.y];
    gl_Position = now_clip;
}
