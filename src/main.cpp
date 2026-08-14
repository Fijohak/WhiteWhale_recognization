#include "main.h"

#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

// GLAD 必须放在 SDL OpenGL 相关头文件之前
#include <glad/gl.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_opengl3.h"

// ==========================================
// App
// ==========================================

#include "app/GroupManager.h"
#include "app/ImageManager.h"

// ==========================================
// Tools
// ==========================================

#include "tools/FolderDialog.h"

// ==========================================
// UI
// ==========================================

#include "ui/MainUi.h"
#include "ui/UiImage.h"


namespace
{
    SDL_Window* window = nullptr;

    SDL_GLContext glContext = nullptr;
}


// =========================================================
// Main
// =========================================================

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    return runApplication();
}


// =========================================================
// Application
// =========================================================

int runApplication()
{
    // =====================================================
    // 1. 初始化 SDL
    // =====================================================

    if (!sdlInit())
    {
        return EXIT_FAILURE;
    }


    // =====================================================
    // 2. 设置 OpenGL 3.2 Core
    // =====================================================

    openglSet();


    // =====================================================
    // 3. 创建窗口
    // =====================================================

    if (!sdlCreateWindow())
    {
        return EXIT_FAILURE;
    }


    // =====================================================
    // 4. 创建 OpenGL Context
    // =====================================================

    if (!opendlContext())
    {
        return EXIT_FAILURE;
    }


    // =====================================================
    // 5. 初始化 GLAD
    //
    // 必须在 OpenGL Context 创建之后
    // =====================================================

    if (!gladInit())
    {
        return EXIT_FAILURE;
    }


    // =====================================================
    // 6. 初始化 Dear ImGui
    // =====================================================

    if (!imguiInit())
    {
        return EXIT_FAILURE;
    }


    // =====================================================
    // 7. Application Objects
    // =====================================================

    MainUi mainUi;

    GroupManager groupManager;

    ImageManager imageManager;

    FolderDialog groupFolderDialog;


    // =====================================================
    // 8. UI Events
    // =====================================================

    UiEvents uiEvents;


    // =====================================================
    // 8.1 加载某一个 Group 的所有图片
    // =====================================================

    auto loadGroup =
        [&](int groupIndex)
    {
        // --------------------------------------
        // 获取 Group
        // --------------------------------------

        const GroupInfo* group =
            groupManager.getGroup(
                groupIndex
            );


        if (group == nullptr)
        {
            imageManager.clear();

            mainUi.clearCompareImages();

            return;
        }


        // --------------------------------------
        // 从 Group Folder 加载图片
        // --------------------------------------

        if (
            !imageManager.loadGroup(
                group->path
            )
        )
        {
            std::cerr
                << "Failed to load group images: "
                << imageManager.getLastError()
                << '\n';


            mainUi.clearCompareImages();

            return;
        }


        // --------------------------------------
        // ImageInfo -> UiImage
        // --------------------------------------

        std::vector<UiImage> uiImages;


        uiImages.reserve(
            imageManager
                .getImages()
                .size()
        );


        for (
            const auto& image :
            imageManager.getImages()
        )
        {
            UiImage uiImage;


            // OpenGL Texture ID
            uiImage.textureId =
                static_cast<ImTextureID>(
                    image.texture.id
                );


            uiImage.width =
                image.texture.width;


            uiImage.height =
                image.texture.height;


            // 暂时不显示文件名。
            //
            // 后续如果需要，可以在这里设置：
            //
            // uiImage.name = ...
            //
            // 目前保持为空。
            uiImage.name.clear();


            uiImages.push_back(
                std::move(
                    uiImage
                )
            );
        }


        // --------------------------------------
        // 更新左侧 ComparePanel
        // --------------------------------------

        mainUi.setCompareImages(
            uiImages
        );


        // --------------------------------------
        // Debug
        // --------------------------------------

        std::cout
            << "Loaded group "
            << groupIndex + 1
            << ", images: "
            << imageManager.getImageCount()
            << '\n';
    };


    // =====================================================
    // 8.2 选择 Group Root Folder
    // =====================================================

    uiEvents.onSelectGroupFolder =
        [&]()
    {
        groupFolderDialog.open(
            window
        );
    };


    // =====================================================
    // 8.3 重新选择 Group Root Folder
    // =====================================================

    uiEvents.onReselectGroupFolder =
        [&]()
    {
        groupFolderDialog.open(
            window,
            groupManager.getRootPath()
        );
    };


    // =====================================================
    // 8.4 点击顶部 Group
    // =====================================================

    uiEvents.onGroupClick =
        [&](int groupIndex)
    {
        loadGroup(
            groupIndex
        );
    };


    // =====================================================
    // 8.5 点击左侧图片
    //
    // 当前只保留事件接口。
    // 真正业务逻辑以后实现。
    // =====================================================

    uiEvents.onImageClick =
        [&](int groupIndex, int imageIndex)
    {
        (void)groupIndex;


        const ImageInfo* image =
            imageManager.getImage(
                imageIndex
            );


        if (image == nullptr)
        {
            return;
        }


        // ======================================
        // TODO:
        //
        // 图片点击事件
        //
        // 当前可获得：
        //
        // groupIndex
        // imageIndex
        // image->path
        //
        // 后续需要什么行为，
        // 再在这里接业务逻辑。
        // ======================================
    };


    // =====================================================
    // 8.6 注册 UI Events
    // =====================================================

    mainUi.setEvents(
        std::move(
            uiEvents
        )
    );


    // =====================================================
    // 9. 程序状态
    // =====================================================

    bool running = true;


    // =====================================================
    // 10. 主循环
    // =====================================================

    while (running)
    {
        // =================================================
        // 10.1 SDL Events
        // =================================================

        SDL_Event event;


        while (SDL_PollEvent(&event))
        {
            // ---------------------------------------------
            // 先让 ImGui 处理输入事件
            // ---------------------------------------------

            ImGui_ImplSDL3_ProcessEvent(
                &event
            );


            // ---------------------------------------------
            // Application Quit
            // ---------------------------------------------

            if (
                event.type ==
                SDL_EVENT_QUIT
            )
            {
                running = false;
            }


            // ---------------------------------------------
            // Window Close
            // ---------------------------------------------

            if (
                event.type ==
                    SDL_EVENT_WINDOW_CLOSE_REQUESTED
                &&
                event.window.windowID ==
                    SDL_GetWindowID(window)
            )
            {
                running = false;
            }
        }


        // =================================================
        // 10.2 Group Folder Dialog Result
        // =================================================

        if (
            auto selectedPath =
                groupFolderDialog.takePath()
        )
        {
            // ---------------------------------------------
            // 扫描 Group Root
            // ---------------------------------------------

            if (
                groupManager.loadRoot(
                    *selectedPath
                )
            )
            {
                // -----------------------------------------
                // 更新顶部 UI
                // -----------------------------------------

                mainUi.setGroupRoot(
                    groupManager.getRootName(),
                    groupManager.getRootPath(),
                    groupManager.getGroupCount()
                );


                // -----------------------------------------
                // 新 Root 已载入。
                //
                // 旧 Group Texture 和 UI 数据清理。
                // -----------------------------------------

                imageManager.clear();

                mainUi.clearCompareImages();


                // -----------------------------------------
                // 自动加载第一个 Group
                // -----------------------------------------

                if (
                    groupManager.getGroupCount()
                    > 0
                )
                {
                    loadGroup(0);
                }


                // =========================================
                // Debug
                // =========================================

                std::cout
                    << '\n'
                    << "=================================="
                    << '\n';


                std::cout
                    << "Group root: "
                    << groupManager.getRootPath()
                    << '\n';


                std::cout
                    << "Group count: "
                    << groupManager.getGroupCount()
                    << '\n';


                std::cout
                    << "=================================="
                    << '\n';


                // -----------------------------------------
                // 打印所有 Group
                // -----------------------------------------

                for (
                    int i = 0;
                    i < groupManager.getGroupCount();
                    ++i
                )
                {
                    const GroupInfo* group =
                        groupManager.getGroup(
                            i
                        );


                    if (group == nullptr)
                    {
                        continue;
                    }


                    std::cout
                        << "Group "
                        << i + 1
                        << ": "
                        << group->name
                        << '\n';


                    std::cout
                        << "    Path: "
                        << group->path
                        << '\n';
                }


                std::cout
                    << "=================================="
                    << '\n'
                    << '\n';
            }
            else
            {
                std::cerr
                    << "Failed to load group root: "
                    << groupManager.getLastError()
                    << '\n';
            }
        }


        // =================================================
        // 10.3 Folder Dialog Error
        // =================================================

        if (
            auto dialogError =
                groupFolderDialog.takeError()
        )
        {
            std::cerr
                << "Folder dialog error: "
                << *dialogError
                << '\n';
        }


        // =================================================
        // 10.4 窗口最小化
        // =================================================

        if (
            SDL_GetWindowFlags(window)
            &
            SDL_WINDOW_MINIMIZED
        )
        {
            SDL_Delay(10);

            continue;
        }


        // =================================================
        // 10.5 Start ImGui Frame
        // =================================================

        ImGui_ImplOpenGL3_NewFrame();

        ImGui_ImplSDL3_NewFrame();

        ImGui::NewFrame();


        // =================================================
        // 11. UI
        // =================================================

        mainUi.draw();


        // =================================================
        // 12. Render
        // =================================================

        ImGui::Render();


        // -------------------------------------------------
        // 获取实际 framebuffer 尺寸
        //
        // Retina / High DPI
        // -------------------------------------------------

        int displayWidth = 0;

        int displayHeight = 0;


        SDL_GetWindowSizeInPixels(
            window,
            &displayWidth,
            &displayHeight
        );


        glViewport(
            0,
            0,
            displayWidth,
            displayHeight
        );


        glClearColor(
            0.08f,
            0.08f,
            0.08f,
            1.0f
        );


        glClear(
            GL_COLOR_BUFFER_BIT
        );


        ImGui_ImplOpenGL3_RenderDrawData(
            ImGui::GetDrawData()
        );


        // =================================================
        // 13. Swap Window
        // =================================================

        SDL_GL_SwapWindow(
            window
        );
    }


    // =====================================================
    // 14. Cleanup
    // =====================================================

    // ImageManager 持有 OpenGL Texture。
    //
    // 必须在销毁 OpenGL Context 之前释放。
    imageManager.clear();


    cleanUp();


    return EXIT_SUCCESS;
}


// =========================================================
// SDL Init
// =========================================================

bool sdlInit()
{
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        std::cerr
            << "SDL_Init failed: "
            << SDL_GetError()
            << '\n';


        return false;
    }


    return true;
}


// =========================================================
// OpenGL Settings
// =========================================================

void openglSet()
{
    // -----------------------------------------------------
    // OpenGL 3.2 Core
    //
    // Windows + macOS 共用
    // -----------------------------------------------------

    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_FLAGS,
        SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG
    );


    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_PROFILE_MASK,
        SDL_GL_CONTEXT_PROFILE_CORE
    );


    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_MAJOR_VERSION,
        3
    );


    SDL_GL_SetAttribute(
        SDL_GL_CONTEXT_MINOR_VERSION,
        2
    );


    SDL_GL_SetAttribute(
        SDL_GL_DOUBLEBUFFER,
        1
    );


    SDL_GL_SetAttribute(
        SDL_GL_DEPTH_SIZE,
        24
    );


    SDL_GL_SetAttribute(
        SDL_GL_STENCIL_SIZE,
        8
    );
}


// =========================================================
// Create SDL Window
// =========================================================

bool sdlCreateWindow()
{
    window =
        SDL_CreateWindow(
            "Image Viewer",
            1280,
            800,

            SDL_WINDOW_OPENGL |
            SDL_WINDOW_RESIZABLE |
            SDL_WINDOW_HIGH_PIXEL_DENSITY
        );


    if (window == nullptr)
    {
        std::cerr
            << "SDL_CreateWindow failed: "
            << SDL_GetError()
            << '\n';


        SDL_Quit();


        return false;
    }


    return true;
}


// =========================================================
// Create OpenGL Context
// =========================================================

bool opendlContext()
{
    glContext =
        SDL_GL_CreateContext(
            window
        );


    if (glContext == nullptr)
    {
        std::cerr
            << "SDL_GL_CreateContext failed: "
            << SDL_GetError()
            << '\n';


        SDL_DestroyWindow(
            window
        );


        SDL_Quit();


        return false;
    }


    // -----------------------------------------------------
    // 设置当前 OpenGL Context
    // -----------------------------------------------------

    if (
        !SDL_GL_MakeCurrent(
            window,
            glContext
        )
    )
    {
        std::cerr
            << "SDL_GL_MakeCurrent failed: "
            << SDL_GetError()
            << '\n';


        SDL_GL_DestroyContext(
            glContext
        );


        SDL_DestroyWindow(
            window
        );


        SDL_Quit();


        return false;
    }


    // -----------------------------------------------------
    // 垂直同步
    // -----------------------------------------------------

    SDL_GL_SetSwapInterval(1);


    return true;
}


// =========================================================
// GLAD Init
// =========================================================

bool gladInit()
{
    const int glVersion =
        gladLoadGL(
            reinterpret_cast<GLADloadfunc>(
                SDL_GL_GetProcAddress
            )
        );


    if (glVersion == 0)
    {
        std::cerr
            << "Failed to initialize GLAD."
            << '\n';


        SDL_GL_DestroyContext(
            glContext
        );


        SDL_DestroyWindow(
            window
        );


        SDL_Quit();


        return false;
    }


    // -----------------------------------------------------
    // Debug OpenGL Info
    // -----------------------------------------------------

    std::cout
        << "GLAD loaded OpenGL "
        << GLAD_VERSION_MAJOR(
               glVersion
           )
        << "."
        << GLAD_VERSION_MINOR(
               glVersion
           )
        << '\n';


    std::cout
        << "OpenGL Version: "
        << reinterpret_cast<const char*>(
               glGetString(
                   GL_VERSION
               )
           )
        << '\n';


    std::cout
        << "OpenGL Renderer: "
        << reinterpret_cast<const char*>(
               glGetString(
                   GL_RENDERER
               )
           )
        << '\n';


    return true;
}


// =========================================================
// Dear ImGui Init
// =========================================================

bool imguiInit()
{
    IMGUI_CHECKVERSION();


    ImGui::CreateContext();


    ImGuiIO& io =
        ImGui::GetIO();


    io.ConfigFlags |=
        ImGuiConfigFlags_NavEnableKeyboard;


    // -----------------------------------------------------
    // Default Theme
    // -----------------------------------------------------

    ImGui::StyleColorsDark();


    // -----------------------------------------------------
    // SDL3 Backend
    // -----------------------------------------------------

    if (
        !ImGui_ImplSDL3_InitForOpenGL(
            window,
            glContext
        )
    )
    {
        std::cerr
            << "ImGui SDL3 backend initialization failed."
            << '\n';


        ImGui::DestroyContext();


        SDL_GL_DestroyContext(
            glContext
        );


        SDL_DestroyWindow(
            window
        );


        SDL_Quit();


        return false;
    }


    // -----------------------------------------------------
    // OpenGL Backend
    //
    // OpenGL 3.2 Core -> GLSL 150
    // -----------------------------------------------------

    if (
        !ImGui_ImplOpenGL3_Init(
            "#version 150"
        )
    )
    {
        std::cerr
            << "ImGui OpenGL backend initialization failed."
            << '\n';


        ImGui_ImplSDL3_Shutdown();


        ImGui::DestroyContext();


        SDL_GL_DestroyContext(
            glContext
        );


        SDL_DestroyWindow(
            window
        );


        SDL_Quit();


        return false;
    }


    return true;
}


// =========================================================
// Cleanup
// =========================================================

void cleanUp()
{
    // -----------------------------------------------------
    // ImGui
    // -----------------------------------------------------

    ImGui_ImplOpenGL3_Shutdown();


    ImGui_ImplSDL3_Shutdown();


    ImGui::DestroyContext();


    // -----------------------------------------------------
    // OpenGL
    // -----------------------------------------------------

    SDL_GL_DestroyContext(
        glContext
    );


    glContext = nullptr;


    // -----------------------------------------------------
    // SDL Window
    // -----------------------------------------------------

    SDL_DestroyWindow(
        window
    );


    window = nullptr;


    // -----------------------------------------------------
    // SDL
    // -----------------------------------------------------

    SDL_Quit();
}
