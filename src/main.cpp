/*
i use this (gcc + ninja)
cmake -S . -B build -DCMAKE_BUILD_TYPE=DEBUG -DCMAKE_CXX_FLAGS='-fdiagnostics-color=always -Wall' -G Ninja
cmake --build build && build/main

        +____________+
        /:\         ,:\
       / : \       , : \
      /  :  \     ,  :  \
     /   :   +-----------+
    +....:../:...+   :  /|
    |\   +./.:...`...+ / |
    | \ ,`/  :   :` ,`/  |
    |  \ /`. :   : ` /`  |
    | , +-----------+  ` |
    |,  |   `+...:,.|...`+
    +...|...,'...+  |   /
     \  |  ,     `  |  /
      \ | ,       ` | /
       \|,         `|/
        +___________+

2-Dimensional Representation Of A 3-Dimensional Cross-Section Of A 4-Dimensional Cube
*/

#include "glad/gl.h"
#include "GLFW/glfw3.h"
#include "GLFW/glfw3native.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "imgui.h"
#include "backends/imgui_impl_opengl3.h"
#include "backends/imgui_impl_glfw.h"
#include "logger.h"
#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

#include "Application.hpp"
#include "utils/ControllableCamera.hpp"
#include "opengl/Framebuffer.hpp"
#include "opengl/UniformBuffer.hpp"
#include "opengl/Cubemap.hpp"
#include "utils/AABB.hpp"
#include "utils/Scene.hpp"

#include <chrono>
#include <memory>
#include <thread>
#include <iostream>
#include <stdexcept>

#ifdef NDEBUG
extern const bool debug = false;
#else
extern const bool debug = true;
#endif

void imguistuff(Application &app);
float lerp(float a, float b, float x) { return a + x * (b - a); }

int main(int argc, char **argv)
{
    Application app; // initialisation
    app.camera = ControllableCamera{app.window, {0, 0, 4}, {-90, 0, 0}};
    app.shaders = {
        {"shaders/raytracing.glsl", true}
    }; // on shader reload contents will be recompiled, if a shader fails it will be restored.

//  =========================================== 

    float vertices[] = {
        // positions        tex coords
        -1.0, -1.0, 0.0,    0.0, 0.0,
         1.0, -1.0, 0.0,    1.0, 0.0,
         1.0,  1.0, 0.0,    1.0, 1.0,
        -1.0,  1.0, 0.0,    0.0, 1.0 
    };
    ShaderProgram HDRshader = ShaderProgram{"shaders/hdr.glsl", true};
    VertexBuffer quadVB = VertexBuffer{vertices, sizeof(vertices)};
    VertexArray quadVA = VertexArray{quadVB, InterleavedVertexBufferLayout{{3, GL_FLOAT}, {2, GL_FLOAT}}};
    
//  =========================================== 

//  =========================================== 

    Texture mainTexture;
    mainTexture.bind();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glClearColor(0, 0, 0, 1);
    constexpr unsigned numPerGroup = 20;
    float focalPlaneDistance = 10;

    std::thread showFps([&app](){ while(!glfwWindowShouldClose(app.window)) { std::this_thread::sleep_for(std::chrono::milliseconds(1000)); glfwSetWindowTitle(app.window, ("lopengl -- " + std::to_string((int) glm::round(1 / app.deltatime)) + " FPS").c_str()); }});

    glm::vec3 wavelengths{200, 400, 700};
    float dencityFalloff = 10;
    glm::vec3 sunPos = {-10, 2, 1};
    glm::vec3 planetRotation = {15, 35, 45};
    glm::vec3 planetPos = {1, 1, 1};
    float planetSize = 3;
    float atmosphereSize = 1;
    float scatteringStrength = 1.0f;
    glm::vec3 scatteringCoefficients;
    bool updateCoefficients = true;
    while (!glfwWindowShouldClose(app.window))
    {
        auto start = std::chrono::high_resolution_clock::now();
        int prevWidth = app.camera.width, prevHeight = app.camera.height;
        glm::vec3 prevCamPos = app.camera.position, prevCamRotation = app.camera.rotation;
        app.camera.update(app.deltatime);
        glfwGetWindowSize(app.window, &app.camera.width, &app.camera.height);
        if(prevWidth != app.camera.width || prevHeight != app.camera.height) { // resize textures
            mainTexture.bind();
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, app.camera.width, app.camera.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            app.numAccumFrames = 0;
        }
        if(prevCamPos != app.camera.position || prevCamRotation != app.camera.rotation) app.numAccumFrames = 0;
//  =========================================== 

        if(updateCoefficients) {
            scatteringCoefficients = {
                glm::pow(300 / wavelengths.r, 4) * scatteringStrength,
                glm::pow(400 / wavelengths.g, 4) * scatteringStrength,
                glm::pow(400 / wavelengths.b, 4) * scatteringStrength
            };
            updateCoefficients = false;
        }

        glViewport(0, 0, app.camera.width, app.camera.height);
        app.shaders[0].bind();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        glBindImageTexture(0, mainTexture.getRenderID(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA16F);
        
        glUniform1i(app.shaders[0].getUniform("u_output"), 0);
        glUniform1ui(app.shaders[0].getUniform("u_numAccumFrames"), app.numAccumFrames);
        glUniform3fv(app.shaders[0].getUniform("u_camera.position"), 1, &app.camera.position.x);
        glUniform3f(app.shaders[0].getUniform("u_camera.forward"), app.camera.getFront().x, app.camera.getFront().y, app.camera.getFront().z);
        glUniform3f(app.shaders[0].getUniform("u_camera.right"), app.camera.getRight().x, app.camera.getRight().y, app.camera.getRight().z);
        glUniform3f(app.shaders[0].getUniform("u_camera.up"), app.camera.getUp().x, app.camera.getUp().y, app.camera.getUp().z);
        glUniform1f(app.shaders[0].getUniform("u_camera.fov"), app.camera.fov);
        glUniform1f(app.shaders[0].getUniform("u_camera.aspect"), (float) app.camera.width / app.camera.height);
        glUniform1f(app.shaders[0].getUniform("u_camera.focalPlaneDistance"), focalPlaneDistance);
        glUniform1f(app.shaders[0].getUniform("u_time"), glfwGetTime());
        glUniform3fv(app.shaders[0].getUniform("u_scatteringCoefficients"), 1, &scatteringCoefficients.r);

        glUniform3fv(app.shaders[0].getUniform("u_planet.position"), 1, &planetPos.x);
        glUniform3f(app.shaders[0].getUniform("u_planet.color"), 0.3, 0.1, 0.01);
        glUniform1f(app.shaders[0].getUniform("u_planet.size"), planetSize);
        glUniform1f(app.shaders[0].getUniform("u_planet.atmosphereSize"), atmosphereSize);
        glUniform1f(app.shaders[0].getUniform("u_planet.atmosphereDencityFalloff"), dencityFalloff);
        glm::mat4 modelMat = glm::rotate(glm::mat4{1.0f}, 1.0f, planetRotation); // only rotate
        glUniformMatrix4fv(app.shaders[0].getUniform("u_planet.invModelMat"), 1, GL_FALSE, &glm::inverse(modelMat)[0][0]);

        glUniform3fv(app.shaders[0].getUniform("u_sun.position"), 1, &sunPos.x);
        glUniform3f(app.shaders[0].getUniform("u_sun.color"), 1, 0.8, 0.2);

        glDispatchCompute(app.camera.width / numPerGroup + 1, app.camera.height / numPerGroup + 1, 1);
        ++app.numAccumFrames;

//  =========================================== 

        HDRshader.bind();
        glUniform1i(HDRshader.getUniform("u_texture"), 0);
        mainTexture.bind(0);
        quadVA.bind();
        quadVB.bind();
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

//  =========================================== 
    ImGuiIO &io = ImGui::GetIO();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::Begin("properties");

    float prevFocalPlaneDistance = focalPlaneDistance;
    ImGui::SliderFloat("dencity falloff", &dencityFalloff, 0, 30);
    ImGui::InputFloat("camera speed", &app.camera.speed);
    ImGui::DragFloat3("sun position", &sunPos.x, 0.5);
    ImGui::InputFloat("atmosphere size", &atmosphereSize);
    if(ImGui::InputFloat("scattering strength", &scatteringStrength)) updateCoefficients = true;
    if(ImGui::DragFloat3("wavelengths", &wavelengths.r, 0.5)) updateCoefficients = true;
    if(prevFocalPlaneDistance != focalPlaneDistance) app.numAccumFrames = 0;

    ImGui::End();
    if(app.failedToReloadShaders) {
        ImGui::OpenPopup("failed to reload shaders!");
        app.failedToReloadShaders = false;
    }
    if(ImGui::BeginPopup("failed to reload shaders!")) {
        ImGui::Text("shader name: %s", app.lastFailedShaderName.c_str());
        ImGui::Separator();
        ImGui::TextWrapped(app.lastFailedShaderLog.c_str());
        ImGui::EndPopup();
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        GLFWwindow *backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }
        
//  =========================================== 
        glfwSwapBuffers(app.window);
        glfwPollEvents();
        ++app.frameCounter;
        app.deltatime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() * 1.0E-6;
    }
    showFps.detach();
}
/*
c++ be like:
Because the lvalueness or rvalueness of an expression is independent of its type, it’s possible to have lvalues whose type is rvalue reference, and it’s also possible to have rvalues of the type rvalue reference.
*/