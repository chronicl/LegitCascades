[declaration: "config"]
{{
}}



[declaration: "line_sweeping"]
{{
  mat2 GetSweepToPixelUniformTransform(vec2 dir, ivec2 size)
  {
    if(abs(dir.x) > abs(dir.y))
    {
      float abs_tan = abs(dir.y / dir.x);
      //https://en.wikipedia.org/wiki/List_of_trigonometric_identities
      float inv_abs_cos = sqrt(1.0f + abs_tan * abs_tan);
      return mat2(dir, vec2(0.0, 1.0) * inv_abs_cos);
    }
    return mat2(1.0f);
  }
  mat2 GetSweepToRectSkewedTransform(vec2 dir, float spacing, ivec2 rect_size)
  {
    if(abs(dir.x) > abs(dir.y))
    {
      return mat2(dir / abs(dir.x) * spacing, vec2(0.0f, spacing));
    }else
    {
      return mat2(dir / abs(dir.y) * spacing, vec2(spacing, 0.0f));
    }
  }
}}

[include: "raymarching", "line_sweeping"]
void TestPass(ivec2 size, sampler2D scene_tex, out vec4 color)
{{
  void main()
  {
    //color = RaymarchInterval(size, gl_FragCoord.xy, vec2(1.0, 0.0), vec2(50.0, 200.0), 12.0, scene_tex);
    mat2 sweep_to_pixel = GetSweepToRectSkewedTransform(vec2(1.0f, 0.9f), 10.0, size);
    mat2 pixel_to_sweep = inverse(sweep_to_pixel);
    vec2 pixel = gl_FragCoord.xy;
    vec2 sweep = pixel_to_sweep * pixel;
    color = vec4(fract(sweep), 0.0, 1.0);
  }
}}


[declaration: "raymarching"]
{{
  vec4 RaymarchInterval(ivec2 size, vec2 ray_start, vec2 ray_dir, vec2 ray_minmax, float step_size, sampler2D scene_tex)
  {
    vec2 inv_size = vec2(1.0f) / vec2(size);
    vec3 radiance = vec3(0.0f);;
    float transmittance = 1.0f;
    for(float ray_scale = ray_minmax.x; ray_scale < ray_minmax.y; ray_scale += step_size)
    {
      vec2 ray_pos = ray_start + ray_dir * ray_scale;
      vec2 uv_pos = ray_pos * inv_size;
      vec4 ray_sample = textureLod(scene_tex, uv_pos, 0.0);
      radiance += ray_sample.rgb * transmittance * ray_sample.a;
      transmittance *= (1.0f - ray_sample.a);
    }
    return vec4(radiance, transmittance);
  }
}}

void ScenePass(ivec2 size, out vec4 radiance)
{{
  vec4 Circle(vec4 prev_radiance, vec2 delta, float radius, vec4 circle_radiance)
  {
    return length(delta) < radius ? circle_radiance : prev_radiance;
  }
  void main()
  {
    radiance = vec4(0.0f);
    radiance = Circle(radiance, vec2(size / 2) - gl_FragCoord.xy, 30.0, vec4(1.0f, 0.5f, 0.0f, 1.0f));
    radiance = Circle(radiance, vec2(size / 2) + vec2(10.0, 0.0) - gl_FragCoord.xy, 30.0, vec4(0.0f, 0.0f, 0.0f, 1.0f));
  }
}}

[rendergraph]
[include: "fps"]
void RenderGraphMain()
{{
  void main()
  {
    Text("Fps: " + GetSmoothFps());
  }
}}

[declaration: "smoothing"]
{{
  float SmoothOverTime(float val, string name, float ratio = 0.95)
  {
    ContextVec2(name) = ContextVec2(name) * ratio + vec2(val, 1) * (1.0 - ratio);
    return ContextVec2(name).x / (1e-7f + ContextVec2(name).y);
  }
}}
  
[declaration: "fps"]
[include: "smoothing"]
{{
  float GetSmoothFps()
  {
    float dt = GetTime() - ContextFloat("prev_time");
    ContextFloat("prev_time") = GetTime();
    ivec2 size = GetSwapchainImage().GetSize();
    Image scene_img = GetImage(size, rgba16f);
    ScenePass(size, scene_img);
    TestPass(size, scene_img, GetSwapchainImage());
    return 1000.0 / (1e-7f + SmoothOverTime(dt, "fps_count"));
  }
}}