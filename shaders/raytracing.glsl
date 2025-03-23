#shader compute
#version 430 core
layout(local_size_x = 20, local_size_y = 20, local_size_z = 1) in;
layout(rgba16f) uniform image2D u_output;

#define debugColor ?vec3(0,1,0):vec3(1,0,0) // output bool values as color
struct Camera {
    float fov;
    float aspect;
    vec3 position;
    vec3 forward;
    vec3 right;
    vec3 up;
    float focalPlaneDistance;
};
struct Ray {
    vec3 direction;
    vec3 origin;
};
struct AABB {
    vec3 min;
    vec3 max;
};
struct Sphere {
    vec3 center;
    float radius;
};
uniform float u_time;
uniform Camera u_camera;

struct Planet {
    vec3 position;
    vec3 color;
    float size;
    float atmosphereSize;
    float atmosphereDencityFalloff;
    mat4 invModelMat; // translate the ray from world space to (rotated) planets local space
};
uniform vec3 u_scatteringCoefficients;
uniform Planet u_planet;
struct Sun {
    vec3 position;
    vec3 color;
};
uniform Sun u_sun;

// intersection tests
vec2 rayAABB(Ray ray, AABB aabb);
bool rayAABBb(Ray ray, AABB aabb);
float raySphere(Ray ray, Sphere sphere);

// random functions
uint rand(inout uint state); // return random uint in [0; 0xffffffffu]
float randNormalDistribution(inout uint state);
float randZeroOne(inout uint state); // return random float in [0; 1]
float randNegOneOne(inout uint state); // return random float in [-1; 1]
vec3 randUnitSphere(inout uint state);
vec3 randHemisphere(inout uint state, vec3 normal);

vec3 rayColor(Ray ray);
Ray calculateRay(vec2 texCoords, Camera camera);
float lerp(float a, float b, float x) { return a + x * (b - a); }
vec3 lerp(vec3 a, vec3 b, float x) { return a + x * (b - a); }
vec2 lerp(vec2 a, vec2 b, float x) { return a + x * (b - a); }

uint seed = 0;
const uint maxBounceCount = 10;
const uint raysPerPixel = 1;

void main() {
    ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
    vec2 numTexels = gl_NumWorkGroups.xy*gl_WorkGroupSize.xy;
    vec2 texCoords = vec2(texelCoord) / numTexels;
    float pixelIndex = texelCoord.y + (numTexels.x + numTexels.y) * texelCoord.x;
    seed = uint(u_time * pixelIndex);

    vec3 color = vec3(0);
    for(uint i = 0; i < raysPerPixel; ++i) {
        color += rayColor(calculateRay(texCoords, u_camera));
    }
    color /= raysPerPixel;
    imageStore(u_output, texelCoord, vec4(color, 1.0));
}

float distanceToCube(vec3 point, vec3 cubeCenter, float cubeSize) {
    vec3 q = abs(cubeCenter - point) - cubeSize;
    float sd = length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
    return max(sd, 0);
}
float dencityAtPoint(vec3 point) {
    float heightAboveSurface = distanceToCube(point, u_planet.position, u_planet.size);
	float height01 = clamp(heightAboveSurface / u_planet.atmosphereSize, 0, 1);
	float localDensity = exp(-height01 * u_planet.atmosphereDencityFalloff) * (1 - height01);
	return localDensity;
}
float opticalDepth(Ray ray, float length) {
    const uint numOpticalDepthPoints = 10;
    vec3 samplePoint = ray.origin;
    float stepSize = length / (numOpticalDepthPoints - 1);
    float opticalDepth = 0;
    for(uint i = 0; i < numOpticalDepthPoints; ++i) {
        float localDencity = dencityAtPoint(samplePoint);
        opticalDepth += localDencity;
        samplePoint += normalize(ray.direction) * stepSize;
    }
    return opticalDepth;
}
vec3 rayColor(Ray ray) {
    Ray newRay; // translate the vector in the planets local space
    newRay.origin = vec3(u_planet.invModelMat * vec4(ray.origin, 1));
    newRay.direction = normalize(vec3(u_planet.invModelMat * vec4(ray.direction, 0)));

    // find intersection points with an atmosphere
    AABB planetAtmosphereBox = AABB(u_planet.position - vec3(u_planet.size + u_planet.atmosphereSize), u_planet.position + vec3(u_planet.size + u_planet.atmosphereSize));
    vec2 atmosphereIntersection = rayAABB(newRay, planetAtmosphereBox);
    if(atmosphereIntersection.x == -1) return vec3(0); // miss
    // find the distance from the camera to the scene (you might use a depth texture or something)
    AABB planetBox = AABB(u_planet.position - vec3(u_planet.size), u_planet.position + vec3(u_planet.size));
    vec2 sceneIntersection = rayAABB(newRay, planetBox); // intersection -- the distance on the ray.

    float distanceTroughAtmosphere = sceneIntersection.x == -1 ?
        atmosphereIntersection.y - atmosphereIntersection.x : 
        sceneIntersection.x - atmosphereIntersection.x;

    // calculate atmosphere color
    const uint numInScatteringPoints = 10; // tweak this
    vec3 inScatterPoint = newRay.origin + normalize(newRay.direction) * atmosphereIntersection.x;
    float stepSize = distanceTroughAtmosphere / (numInScatteringPoints - 1);
    vec3 inScatteredLight = vec3(0);
    float viewRayOpticalDepth = 0;
    for(uint i = 0; i < numInScatteringPoints; ++i) {
        vec3 dirToSun = normalize(u_sun.position - inScatterPoint);
        vec2 sunRayAtmosphereIntersection = rayAABB(Ray(dirToSun, u_sun.position), planetAtmosphereBox);
        float sunRayLength = length(u_sun.position + sunRayAtmosphereIntersection.x * dirToSun - inScatterPoint);

        float sunRayOpticalDepth = opticalDepth(Ray(-dirToSun, inScatterPoint), sunRayLength);
        viewRayOpticalDepth = opticalDepth(Ray(newRay.direction, inScatterPoint), stepSize * i);

        vec3 transmittance = exp(-(sunRayOpticalDepth + viewRayOpticalDepth) * u_scatteringCoefficients);
        inScatteredLight += dencityAtPoint(inScatterPoint) * transmittance * u_scatteringCoefficients * stepSize;
        inScatterPoint += normalize(ray.direction) * stepSize;
    }
    vec3 originalColor = lerp(vec3(0), u_planet.color, float(sceneIntersection.x != -1)); // change this

    float originalColorTransmittance = exp(-viewRayOpticalDepth);
    inScatteredLight += originalColor * originalColorTransmittance;

    return originalColor * (1 - inScatteredLight) + inScatteredLight;

}
Ray calculateRay(vec2 texCoords, Camera camera) {
    vec2 NDCcoords = texCoords * 2.0 - 1.0;
    float nearPlaneScale = tan(radians(camera.fov) * 0.5);
    vec2 viewPortCoords = vec2(NDCcoords.x * camera.aspect * nearPlaneScale, NDCcoords.y * nearPlaneScale) + vec2(randNegOneOne(seed), randNegOneOne(seed)) * 0.0001;

    vec3 rayDir = camera.forward + viewPortCoords.x * camera.right + viewPortCoords.y * camera.up;
    vec3 rayOrigin = camera.position;
    // vec3 focalPlanePoint = rayOrigin + normalize(rayDir) * camera.focalPlaneDistance;
    // rayOrigin += (randNegOneOne(seed) * camera.right + randNegOneOne(seed) * camera.up) * 0.5;
    // rayDir = focalPlanePoint - rayOrigin;

    return Ray(normalize(rayDir), rayOrigin);
}
vec2 rayAABB(Ray ray, AABB aabb) {
    vec3 rayInvDir = 1 / ray.direction;
    vec3 tMin = (aabb.min - ray.origin) * rayInvDir;
    vec3 tMax = (aabb.max - ray.origin) * rayInvDir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);

    return lerp(
        vec2(-1, -1), 
        lerp(
            vec2(tNear, tFar),
            vec2(0, tFar),
            float(tNear < 0)
        ), 
        // vec2(tNear, tFar),
        float(tFar >= tNear && tFar > 0)
    ); // wut
}
bool rayAABBb(Ray ray, AABB aabb) {
    vec3 rayInvDir = 1 / ray.direction;
    vec3 tMin = (aabb.min - ray.origin) * rayInvDir;
    vec3 tMax = (aabb.max - ray.origin) * rayInvDir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);

    return tFar > tNear && tFar > 0;
}
float raySphere(Ray ray, Sphere sphere) {
    const vec3 OC = sphere.center - ray.origin;
    const float a = dot(ray.direction, ray.direction);
    const float b = -2.0 * dot(ray.direction, OC);
    const float c = dot(OC, OC) - sphere.radius * sphere.radius;
    const float discriminant = b*b - 4*a*c;
    if(discriminant < 0) return -1;
    
    const float sqrtDiscriminant = sqrt(discriminant);
    float t = min((-b - sqrtDiscriminant) / (2*a), (-b + sqrtDiscriminant) / (2*a));

    if(t < 0) return -1;

    return t;
}

uint rand(inout uint state) {
	state = state * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}
float randZeroOne(inout uint state) {
    return float(rand(state)) * (1.0 / float(0xffffffffu));
}
float randNegOneOne(inout uint state) {
    return randZeroOne(state) * 2.0 - 1.0;
}
vec3 randUnitSphere(inout uint state) {
    return normalize(vec3(randNormalDistribution(state) * 2 - 1, randNormalDistribution(state) * 2 - 1, randNormalDistribution(state))) * 2 - 1;
}
vec3 randHemisphere(inout uint state, vec3 normal) {
    vec3 randSphere = randUnitSphere(state);

    if(dot(normal, randSphere) < 0) return -randSphere;
    return randSphere;
}
float randNormalDistribution(inout uint state) {
    // Thanks to https://stackoverflow.com/a/6178290
    float theta = 2 * 3.1415926 * randZeroOne(state);
    float rho = sqrt(-2 * log(randZeroOne(state)));
    return rho * cos(theta);
}