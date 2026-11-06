// The motion at one render pixel (camera_motion.comp's, which explains it):
// where the surface in it was a frame ago minus where it is, in render
// pixels, from its raw depth, the two cameras (the Params block both
// shaders open with), the characters' capture (cap_map) and the game's
// velocity map (velo_map). Shared by the motion pass and TAA
// (taa_resolve.comp works it out for its own tile and needs no motion pass).
// The including shader defines MOTION_SAMPLER.
vec2 motion_at(const ivec2 xy, const float raw) {
    const vec2 size = prev_proj.zw;
    vec2 mv = vec2(0.0);
    if (prev_proj.x > 0.0) {
        const vec2 uv = (vec2(xy) + 0.5) / size;
        const vec2 ndc = uv * 2.0 - 1.0;
        const vec3 dir = vec3(ndc.x * now_proj.x, -ndc.y * now_proj.y, 1.0);
        const float d = now_proj.w / (raw - now_proj.z);
        vec3 gp;
        if (d > 0.0 && d < 100000.0) {
            const vec4 g = vec4(dir * d, 1.0);
            const vec4 w = vec4(dot(now_r0, g), dot(now_r1, g), dot(now_r2, g), 1.0);
            gp = vec3(dot(prev_r0, w), dot(prev_r1, w), dot(prev_r2, w));
        } else {
            // The sky: the direction turned, no translation.
            const vec3 w = vec3(dot(now_r0.xyz, dir), dot(now_r1.xyz, dir), dot(now_r2.xyz, dir));
            gp = vec3(dot(prev_r0.xyz, w), dot(prev_r1.xyz, w), dot(prev_r2.xyz, w));
        }
        if (gp.z > 0.0001) {
            const vec2 prev_uv = vec2((gp.x / (prev_proj.x * gp.z) + 1.0) * 0.5, (-gp.y / (prev_proj.y * gp.z) + 1.0) * 0.5);
            mv = clamp((prev_uv - uv) * size, vec2(-60000.0), vec2(60000.0));
            if (any(isnan(mv))) mv = vec2(0.0);
        }
        // The map's mode (velo_size.z): 1 the game's encoding, the bones' motion
        // alone (added to the camera's); 4 the same encoding with the pass
        // given last frame's matrices (render.cpp velo_previous_locked): the
        // whole motion; 2 the map everywhere (a test); 3 rebuilt; 5 rebuilt
        // with each mesh fragment's view distance, tested against the depth.
        const int mode = int(velo_size.z + 0.5);
        bool captured = false;
        if (d > 0.0 && d < 100000.0) {
            const vec4 c = texelFetch(sampler2D(cap_map, MOTION_SAMPLER), xy, 0);
            if (abs(c.a - 0.5) < 0.1 && c.b <= d * 1.004 + 0.01 && d - c.b <= d * 0.03 + 0.1) {
                const vec2 own = vec2(-0.5 * c.r, 0.5 * c.g) * size;
                if (!any(isnan(own)) && !any(isinf(own)) && all(lessThan(abs(own), size))) {
                    mv = own;
                    captured = true;
                }
            }
        }
        if (captured) {
        } else if (mode == 3 || mode == 5) {
            const float seen_d = d;  // the seen surface's view distance (d below is the map's delta)
            const ivec2 vxy = clamp(ivec2(uv * velo_size.xy), ivec2(0), ivec2(velo_size.xy) - 1);
            const vec4 t = texelFetch(sampler2D(velo_map, MOTION_SAMPLER), vxy, 0);
            // Mode 5: the map's blue holds the mesh fragment's view distance,
            // and its motion is taken only where that fragment is the surface
            // seen - within 0.4 % + 1 cm of the depth's, or in front of it by
            // up to a fur's depth (3 % + 10 cm: strands over the body they lie
            // on) - BBNextGenSuite's test. Elsewhere the motion of what is seen
            // stays: the wall through a cape's strands, a fence before a
            // character, the sky past an edge.
            const bool seen = mode != 5 || (seen_d > 0.0 && seen_d < 100000.0 && t.b <= seen_d * 1.004 + 0.01 &&
                                            seen_d - t.b <= seen_d * 0.03 + 0.1);
            if (abs(t.a - 0.5) < 0.1 && seen) {
                const vec2 d = (t.rg - velo_k.zw) * velo_k.xy;
                const vec2 own = vec2(-0.5 * d.x, 0.5 * d.y) * size;
                // Whole (velo_size.w: the pass given last frame's matrices),
                // else the bones' alone, added to the camera's. Over a screen
                // a frame: not motion.
                if (!any(isnan(own)) && !any(isinf(own)) && all(lessThan(abs(own), size))) mv = velo_size.w > 0.5 ? own : mv + own;
            }
        } else if (mode > 0) {
            const ivec2 vxy = clamp(ivec2(uv * velo_size.xy), ivec2(0), ivec2(velo_size.xy) - 1);
            const vec2 raw = texelFetch(sampler2D(velo_map, MOTION_SAMPLER), vxy, 0).rg;
            const vec2 v = raw - 0.5;
            // What the camera's pass wrote for this motion: (uv now - uv then) x its scale.
            const vec2 cam = -mv / size * velo_k.xy;
            if (mode == 2 || any(greaterThan(abs(v - cam), vec2(velo_size.w)))) {
                const vec2 own = -v / velo_k.zw * size;
                // The encoding is clamped to 0..1: a whole motion past it is
                // not held, and the camera's alone stands there.
                const bool clamped = any(lessThan(raw, vec2(0.0005))) || any(greaterThan(raw, vec2(0.9995)));
                if (!any(isnan(own)) && !any(isinf(own)) && !(mode == 4 && clamped)) mv = mode == 1 ? mv + own : own;
            }
        }
    }
    return mv;
}
