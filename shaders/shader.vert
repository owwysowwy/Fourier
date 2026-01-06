#version 410 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 iPos;
layout (location = 2) in float iRadius;
layout (location = 3) in float iAngle;
layout (location = 4) in float iFreq;

uniform mat4 projection;
uniform float time;

void main()
{
    float angle = iAngle + iFreq * time;

    mat2 rot = mat2(
        cos(angle), sin(angle),
        -sin(angle), cos(angle)
    );

    vec2 rotated = vec2(rot * aPos.xy);

    vec3 world = vec3(rotated * iRadius, 0) + iPos;
    gl_Position = projection * vec4(world, 1.0);
}
