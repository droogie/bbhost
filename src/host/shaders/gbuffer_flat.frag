#version 450
// A plugin's draw into the G-buffer (kGxDrawGbuffer, plugin API 19), in place
// of the draw's own pixel shader: the game's lighting then lights it as it
// does the map - the sun and its shadows, the lamps, the fog.
//
// Its one input is the vertex colour as the game's simple coloured draw
// (VS 4b8827a6) passes it on, untouched: rgb the albedo (linear), alpha the
// face's world-space normal when negative - -(1 + u + 2048 v), u and v its
// octahedral coordinates in 11 bits each, a whole number a float holds
// exactly, read flat so no interpolation rounds it. Any other alpha: up.
//
// The targets as the map's own shader (6a833138+bb10e440) writes them:
// 0 the normal (n * 0.5 + 0.5), 1 gloss / wetness / baked shadow, 2 the
// albedo (an sRGB target: linear in), 3 sqrt of the specular colour, 4 the
// indirect diffuse light (the map's lightmap; here a constant ambient), 5 the
// baked specular. Target 6 is not written (masked: fixed_state).
//
// Compiled offline (tools/build_shaders.sh) into gbuffer_flat_frag.spv.h.
layout(location = 0) flat in vec4 colour;
layout(location = 0) out vec4 rt0;
layout(location = 1) out vec4 rt1;
layout(location = 2) out vec4 rt2;
layout(location = 3) out vec4 rt3;
layout(location = 4) out vec4 rt4;
layout(location = 5) out vec4 rt5;

const float kGloss = 0.05;     // matte
const float kSpecular = 0.04;  // a plain dielectric
// The indirect light (target 4) in place of a lightmap: the host writes its
// level into the module (gbuffer_ambient_words) before making it.
layout(constant_id = 0) const float kAmbient = 0.04;
// Target 3's alpha: the light and fog group / 255 - the instance's byte in
// the map's shader, which the lighting reads back as an index into the
// frame's cbLight and cbFog; the game numbers them afresh each frame, so the
// host writes the one its own G-buffer draws use this frame
// (render.cpp gbuffer_group_take). Target 5's: a 2-bit per-instance value, 0.
layout(constant_id = 1) const float kRt3Alpha = 0.0;
layout(constant_id = 2) const float kRt5Alpha = 0.0;

vec3 octahedral(vec2 e) {
    vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

void main() {
    vec3 n = vec3(0.0, 1.0, 0.0);
    if (colour.a < 0.0) {
        const uint code = uint(-colour.a - 1.0 + 0.5);
        n = octahedral(vec2(code & 2047u, code >> 11) / 2047.0 * 2.0 - 1.0);
    }
    // The face's other side seen (the collision is one-sided, and the camera
    // goes behind it): its normal turned to the viewer, as the map's shaders
    // turn theirs, else the lights shade it from behind. Lo-Hit's winding
    // puts the faces it sees from their front on the rasterizer's back (a
    // floor from above is not front-facing).
    if (gl_FrontFacing) n = -n;
    rt0 = vec4(n * 0.5 + 0.5, 0.0);
    rt1 = vec4(kGloss, 0.0, 1.0, 0.0);
    rt2 = vec4(colour.rgb, 1.0);
    rt3 = vec4(vec3(sqrt(kSpecular)), kRt3Alpha);
    rt4 = vec4(vec3(kAmbient), 1.0);
    rt5 = vec4(vec3(0.0), kRt5Alpha);
}
