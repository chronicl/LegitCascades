[declaration: "config"]
{{
  #define POS_FIRST_LAYOUT 1
  const float pi = 3.141592f;
}}

[blendmode: alphablend]
void OverlayTexShader(
  sampler2D tex,
  out vec4 color)
{{
  void main()
  {
    uvec2 pixel_idx = uvec2(gl_FragCoord.xy);
    uvec2 tex_size = uvec2(textureSize(tex, 0));
    if(pixel_idx.x < tex_size.x && pixel_idx.y < tex_size.y)
    {
      color = vec4(texelFetch(tex, ivec2(pixel_idx), 0).rgb, 1.0);
    }else
    {
      color = vec4(0.0);
    }
  }
}}

[include: "atlas_layout", "probe_atlas", "pcg", "raymarching", "probe_regular_grid", "bilinear_interpolation"]
void RaymarchAtlasShader(
  uvec2 size,
  uvec2 c0_size,
  int cascade_scaling_pow2,
  uint cascades_count,
  uint dir_scaling,
  uvec2 c0_probe_size,
  sampler2D scene_tex,
  sampler2D prev_atlas_tex,
  out vec4 color)
{{
  vec4 CastMergedIntervalBilinearFix(
    vec2 screen_pos,
    vec2 dir,
    vec2 interval_minmax,
    ProbeLayout prev_probe_layout,
    GridTransform prev_probe_to_screen,
    uint prev_cascade_idx,
    uint prev_dir_idx)
  {
    GridTransform screen_to_prev_probe = GetInverseTransform(prev_probe_to_screen);
    vec2 prev_probe_idx = ApplyTransform(screen_to_prev_probe, screen_pos);

    BilinearSamples bilinear_samples = GetBilinearSamples(prev_probe_idx);
    vec4 weights = GetBilinearWeights(bilinear_samples.ratio);

    vec4 merged_interval = vec4(0.0f);
    for(uint i = 0u; i < 4u; i++)
    {
        ivec2 prev_probe_idx = clamp(bilinear_samples.base_idx + GetBilinearOffset(i), ivec2(0), ivec2(prev_probe_layout.count) - ivec2(1));
        prev_probe_location.dir_index = prev_dir_index;


        int pixel_index = ProbeLocationToPixelIndex(prev_probe_location, c0_size);
        ivec3 texel_index = PixelIndexToCubemapTexel(face_size, pixel_index);

        vec4 prev_interval = vec4(0.0f, 0.0f, 0.0f, 1.0f);
        if(prev_cascade_index < nCascades)
            prev_interval = cubemapFetch(iChannel0, texel_index.z, texel_index.xy);

        vec2 prev_screen_pos = GetProbeScreenPos(vec2(prev_probe_location.probe_index), prev_probe_location.cascade_index, c0_size);

        vec2 ray_start = screen_pos * vec2(viewport_size) + dir * interval_length.x;
        vec2 ray_end = prev_screen_pos * vec2(viewport_size) + dir * interval_length.y;                

        RayHit ray_hit = radiance(iChannel1, ray_start, normalize(ray_end - ray_start), length(ray_end - ray_start));
        merged_interval += MergeIntervals(ray_hit.radiance, prev_interval) * weights[i];
    }
    return merged_interval;*/
    return vec4(0.0f);
  }

  void main()
  {
    uvec2 atlas_texel_idx = uvec2(gl_FragCoord.xy);

    uint c0_dirs_count = uint(c0_probe_size.x * c0_probe_size.y);
    uvec2 atlas_size = GetAtlasSize(cascade_scaling_pow2, uint(cascades_count), uvec2(c0_size));

    AtlasTexelLocation loc = GetAtlasPixelLocationPosFirst(
      atlas_texel_idx,
      cascade_scaling_pow2,
      uvec2(c0_probe_size),
      uint(dir_scaling),
      uint(cascades_count),
      uvec2(c0_size));
    uint dirs_count = loc.probe_layout.size.x * loc.probe_layout.size.y;

    uvec2 dir_idx2 = uvec2(loc.dir_idx % loc.probe_layout.size.x, loc.dir_idx / loc.probe_layout.size.x);
    if(
      loc.probe_idx.x < loc.probe_layout.count.x &&
      loc.probe_idx.y < loc.probe_layout.count.y &&
      loc.dir_idx < dirs_count)
    {
      vec2 c0_probe_spacing = GetC0ProbeSpacing(size, loc.c0_probe_layout.count);
      vec2 probe_spacing = GetProbeSpacing(c0_probe_spacing, loc.cascade_idx, loc.probe_scaling.spacing_scaling);
      GridTransform probe_to_screen = GetProbeToScreenTransform(probe_spacing);

      vec2 ray_start = ApplyTransform(probe_to_screen, vec2(loc.probe_idx));
      float ang = 2.0f * pi * (float(loc.dir_idx) + 0.5f) / float(dirs_count);

      vec2 ray_dir = vec2(cos(ang), sin(ang));
      vec2 ray_minmax = vec2(0.0f, 100.0f);
      float step_size = 3.0f;
      vec4 radiance = RaymarchInterval(
        size, ray_start, ray_dir, ray_minmax, step_size, scene_tex);

      color = radiance;
      /*vec3 probe_col = hash3i3f(ivec3(loc.probe_idx, 0));
      vec3 dir_col = hash3i3f(ivec3(loc.dir_idx, 0, 0));
      vec3 cascade_col = hash3i3f(ivec3(loc.cascade_idx, 1, 0));
      color = vec4(mix(dir_col, cascade_col, 0.8f), 1.0);*/
    }else
    {
      if(atlas_texel_idx.x < atlas_size.x && atlas_texel_idx.y < atlas_size.y)
      {
        color = vec4(0.2, 0.0, 0.0, 1.0);
      }else
      {
        color = vec4(0.0, 0.0, 0.0, 1.0);
      }
    }
  }
}}


[rendergraph]
[include: "fps", "atlas_layout"]
void RenderGraphMain()
{{
  void main()
  {    
    uvec2 size = GetSwapchainImage().GetSize();
    ClearShader(GetSwapchainImage());
    uvec2 c0_size;
    c0_size.x = SliderInt("c0_size.x", 1, 1024, 512);
    c0_size.y = SliderInt("c0_size.y", 1, 1024, 128);
    int cascade_scaling_pow2 = SliderInt("cascade_scaling_pow2", -1, 1, 0);
    uint cascades_count = SliderInt("cascades count", 1, 10, 4);
    uint dir_scaling = SliderInt("dir_scaling", 1, 10, 4);
    uvec2 c0_probe_size = uvec2(SliderInt("c0_probe_size", 1, 100, 10));

    Image scene_img = GetImage(size, rgba16f);
    SceneShader(size, scene_img);

    uvec2 atlas_size = GetAtlasSize(cascade_scaling_pow2, cascades_count, c0_size);

    Image merged_atlas_img = GetImage(atlas_size, rgba16f);
    Image prev_atlas_img = GetImage(atlas_size, rgba16f);
    RaymarchAtlasShader(
      size,
      c0_size,
      cascade_scaling_pow2,
      cascades_count,
      dir_scaling,
      c0_probe_size,
      scene_img,
      prev_atlas_img,
      merged_atlas_img
    );
    CopyShader(merged_atlas_img, prev_atlas_img);

    OverlayTexShader(
      merged_atlas_img,
      GetSwapchainImage());

    Text("Fps: " + GetSmoothFps());
  }
}}

void ClearShader(out vec4 col)
{{
  void main()
  {
    col = vec4(0.0f, 0.0f, 0.0f, 1.0f);
  }
}}

void CopyShader(sampler2D tex, out vec4 col)
{{
  void main()
  {
    col = texelFetch(tex, ivec2(gl_FragCoord.xy), 0);
  }
}}
[declaration: "atlas_layout"]
{{
  uvec2 GetAtlasSize(int cascade_scaling_pow2, uint cascades_count, uvec2 c0_size)
  {
    if(cascade_scaling_pow2 == 0) //constant size
    {
      return uvec2(c0_size.x, c0_size.y * cascades_count);
    }else
    if(cascade_scaling_pow2 == -1) //cascades get 2x smaller
    {
      return uvec2(c0_size.x, c0_size.y * uint(2));
    }else //cascades get 2x larger
    {
      return uvec2(c0_size.x, c0_size.y * ((uint(1) << cascades_count) - uint(1)));
    }
  }
}}

[declaration: "grid_transform"]
{{
  struct GridTransform
  {
    vec2 spacing;
    vec2 origin;
  };
  GridTransform GetGridTransform(vec2 src_grid_spacing, vec2 src_grid_origin)
  {
    return GridTransform(src_grid_spacing, src_grid_origin);
  }
  GridTransform GetInverseTransform(GridTransform transform)
  {
    //dst = src * xy + zw
    //src = (dst - zw) / xy
    vec2 inv_spacing = vec2(1.0) / transform.spacing;
    return GridTransform(inv_spacing, -transform.origin * inv_spacing);
  }
  GridTransform CombineTransform(GridTransform src_to_tmp, GridTransform tmp_to_dst)
  {
    return GridTransform(src_to_tmp.spacing * tmp_to_dst.spacing, src_to_tmp.origin + tmp_to_dst.spacing + tmp_to_dst.origin);
  }
  GridTransform GetSrcToDstTransform(GridTransform src_transform, GridTransform dst_transform)
  {
    return CombineTransform(src_transform, GetInverseTransform(dst_transform));
  }
  vec2 ApplyTransform(GridTransform transform, vec2 p)
  {
    return transform.spacing * p + transform.origin;
  }
}}
[declaration: "probe_regular_grid"]
[include: "grid_transform"]
{{
  vec2 GetC0ProbeSpacing(uvec2 size, uvec2 c0_probes_count)
  {
    return vec2(size) / vec2(c0_probes_count);
  }
  vec2 GetProbeSpacing(vec2 c0_probe_spacing, uint cascade_idx, vec2 probe_spacing_scaling)
  {
    return c0_probe_spacing * pow(probe_spacing_scaling, vec2(cascade_idx));
  }
  GridTransform GetProbeToScreenTransform(vec2 probe_spacing)
  {
    return GridTransform(probe_spacing, probe_spacing * 0.5f);
  }
}}

[include: "config"]
[declaration: "probe_atlas"]
{{
  struct CascadeLayout
  {
    uvec2 size;
    uvec2 offset;
  };
  CascadeLayout GetCascadeLayout(int cascade_scaling_pow2, uint cascade_idx, uvec2 c0_size)
  {
    CascadeLayout cascade_layout;
    if(cascade_scaling_pow2 == 0) //constant size
    {
      cascade_layout.size = c0_size;
      cascade_layout.offset = uvec2(0u, c0_size.y * cascade_idx);
    }else
    if(cascade_scaling_pow2 == -1) //cascades get 2x smaller
    {
      cascade_layout.size = uvec2(c0_size.x, c0_size.y >> cascade_idx);
      cascade_layout.offset = uvec2(0u, (c0_size.y - (c0_size.y >> cascade_idx)) * 2u);
    }else //cascades get 2x larger
    {
      cascade_layout.size = uvec2(c0_size.x, c0_size.y << cascade_idx);
      cascade_layout.offset = uvec2(0u, c0_size.y * ((1u << cascade_idx) - 1u));
    }
    return cascade_layout;
  }

  struct ProbeScaling
  {
    vec2 spacing_scaling;
    vec2 size_scaling;
  };

  ProbeScaling GetProbeScaling(
    int cascade_scaling_pow2,
    uint dirs_scaling)
  {
    ProbeScaling probe_scaling;
    if(cascade_scaling_pow2 == 0) //constant size cascades
    {
      uint spacing_scaling_sqr = dirs_scaling;
      probe_scaling.spacing_scaling = vec2(sqrt(float(spacing_scaling_sqr)));
      probe_scaling.size_scaling = probe_scaling.spacing_scaling;
    }else
    if(cascade_scaling_pow2 == -1) //cascades get 2x smaller
    {
      uint spacing_scaling_sqr = dirs_scaling * 2u;
      probe_scaling.spacing_scaling = vec2(sqrt(float(spacing_scaling_sqr)));
      probe_scaling.size_scaling = probe_scaling.spacing_scaling * vec2(1.0f, 0.5f);
    }else //cascades get 2x larger
    {
      float spacing_scaling_sqr = float(dirs_scaling) * 0.5f;
      probe_scaling.spacing_scaling = vec2(sqrt(float(spacing_scaling_sqr)));
      probe_scaling.size_scaling = probe_scaling.spacing_scaling * vec2(1.0f, 2.0f);
    }
    return probe_scaling;
  }
  struct ProbeLayout
  {
    uvec2 count;
    uvec2 size;
  };
  ProbeLayout GetProbeLayout(
    uint cascade_idx,
    ProbeLayout c0_probe_layout,
    ProbeScaling probe_scaling,
    uvec2 cascade_size)
  {
    float cascade_idxf = float(cascade_idx);
    vec2 spacing_mult = pow(probe_scaling.spacing_scaling, vec2(cascade_idxf));
    vec2 size_mult = pow(probe_scaling.size_scaling, vec2(cascade_idxf));
    float eps = 1e-5f;
    ProbeLayout probe_layout;
    probe_layout.size = uvec2(floor(vec2(c0_probe_layout.size) * size_mult + vec2(eps)));
    probe_layout.count = cascade_size / probe_layout.size;
    return probe_layout;
  }

  struct AtlasTexelLocation
  {
    ProbeLayout c0_probe_layout;

    uint cascade_idx;
    uvec2 probe_idx;
    uint dir_idx;

    ProbeScaling probe_scaling;
    ProbeLayout probe_layout;
  };

  AtlasTexelLocation GetAtlasTexelLocationDirFirst(uvec2 atlas_texel, uint cascades_count)
  {
    AtlasTexelLocation loc;

    return loc;
  }
  AtlasTexelLocation GetAtlasPixelLocationPosFirst(
    uvec2 atlas_texel,
    int cascade_scaling_pow2,
    uvec2 c0_probe_size,
    uint dir_scaling,
    uint cascades_count,
    uvec2 c0_size)
  {
    CascadeLayout cascade_layout;

    AtlasTexelLocation loc;
    for(loc.cascade_idx = 0u; loc.cascade_idx < cascades_count; loc.cascade_idx++)
    {
      cascade_layout = GetCascadeLayout(cascade_scaling_pow2, loc.cascade_idx, c0_size);
      if(atlas_texel.y >= cascade_layout.offset.y && atlas_texel.y < cascade_layout.offset.y + cascade_layout.size.y)
      {
        break;
      }
    }

    loc.c0_probe_layout.size = c0_probe_size;
    loc.c0_probe_layout.count = c0_size / loc.c0_probe_layout.size;

    loc.probe_scaling = GetProbeScaling(cascade_scaling_pow2, dir_scaling);
    loc.probe_layout = GetProbeLayout(loc.cascade_idx, loc.c0_probe_layout, loc.probe_scaling, cascade_layout.size);
    
    uvec2 cascade_texel = atlas_texel - cascade_layout.offset;

    #if(POS_FIRST_LAYOUT)
      uvec2 cells_count = loc.probe_layout.size;
      uvec2 cell_size = cascade_layout.size / cells_count;
      loc.probe_idx = cascade_texel % cell_size;
      uvec2 dir_idx2 = cascade_texel / cell_size;
      if(dir_idx2.x < loc.probe_layout.size.x && dir_idx2.y < loc.probe_layout.size.y)
        loc.dir_idx = dir_idx2.x + dir_idx2.y * loc.probe_layout.size.x;
      else
        loc.dir_idx = loc.probe_layout.size.x * loc.probe_layout.size.y + 1u;
    #else
      loc.probe_idx = cascade_texel / loc.probe_layout.size;
      uvec2 dir_idx2 = cascade_texel % loc.probe_layout.size;
      loc.dir_idx = dir_idx2.x + dir_idx2.y * loc.probe_layout.size.x;
    #endif

    return loc;
  }
}}


[declaration: "raymarching"]
{{
  vec4 RaymarchInterval(uvec2 size, vec2 ray_start, vec2 ray_dir, vec2 ray_minmax, float step_size, sampler2D scene_tex)
  {
    vec2 inv_size = vec2(1.0f) / vec2(size);
    vec3 radiance = vec3(0.0f);
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

void SceneShader(uvec2 size, out vec4 radiance)
{{
  vec4 Circle(vec4 prev_radiance, vec2 delta, float radius, vec4 circle_radiance)
  {
    return length(delta) < radius ? circle_radiance : prev_radiance;
  }
  void main()
  {
    radiance = vec4(0.0f);
    radiance = Circle(radiance, vec2(size / 2u) - gl_FragCoord.xy, 30.0, vec4(1.0f, 0.5f, 0.0f, 1.0f));
    radiance = Circle(radiance, vec2(size / 2u) + vec2(10.0, 0.0) - gl_FragCoord.xy, 30.0, vec4(0.1f, 0.1f, 0.1f, 1.0f));
  }
}}

[declaration: "pcg"]
{{
  //http://www.jcgt.org/published/0009/03/02/paper.pdf
  uvec3 hash33UintPcg(uvec3 v)
  {
      v = v * 1664525u + 1013904223u;
      v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
      //v += v.yzx * v.zxy; //swizzled notation is not exactly the same because components depend on each other, but works too

      v ^= v >> 16u;
      v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
      //v += v.yzx * v.zxy;
      return v;
  }

  vec3 hash3i3f(ivec3 seed)
  {
      uvec3 hash_uvec3 = hash33UintPcg(uvec3(seed));
      return vec3(hash_uvec3) * (1.0f / float(~0u));
  }
}}

[declaration: "bilinear_interpolation"]
{{
  struct BilinearSamples
  {
      ivec2 base_idx;
      vec2 ratio;
  };

  vec4 GetBilinearWeights(vec2 ratio)
  {
      return vec4(
          (1.0f - ratio.x) * (1.0f - ratio.y),
          ratio.x * (1.0f - ratio.y),
          (1.0f - ratio.x) * ratio.y,
          ratio.x * ratio.y);
  }

  ivec2 GetBilinearOffset(uint offset_index)
  {
      ivec2 offsets[4] = ivec2[4](ivec2(0, 0), ivec2(1, 0), ivec2(0, 1), ivec2(1, 1));
      return offsets[offset_index];
  }
  BilinearSamples GetBilinearSamples(vec2 pixel_idx2f)
  {
      BilinearSamples samples;
      samples.base_idx = ivec2(floor(pixel_idx2f));
      samples.ratio = fract(pixel_idx2f);
      return samples;
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

    return 1000.0 / (1e-7f + SmoothOverTime(dt, "fps_count"));
  }
}}