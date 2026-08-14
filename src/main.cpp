#include "main.h"

#include <cstdlib>
#include <iostream>
#include <string>
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
#include "app/ProcessManager.h"

// ==========================================
// Tools
// ==========================================

#include "tools/FileDialog.h"
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


    // 左侧 Group 数据
    GroupManager groupManager;


    // 左侧 Group 图片
    ImageManager imageManager;


    // 右侧 Single / Batch 图片状态
    ProcessManager processManager;


    // 顶部 Group Root 目录选择器
    FolderDialog groupFolderDialog;


    // 右侧 Batch 目录选择器
    FolderDialog batchFolderDialog;


    // 右侧 Single 文件选择器
    FileDialog singleFileDialog;


    // =====================================================
    // 8. UI Events
    // =====================================================

    UiEvents uiEvents;


    // =====================================================
    // 8.1 ImageTexture -> UiImage
    // =====================================================

    auto makeUiImage =
        [](const ImageTexture& texture)
    {
        UiImage uiImage;


        uiImage.textureId =
            static_cast<ImTextureID>(
                texture.id
            );


        uiImage.width =
            texture.width;


        uiImage.height =
            texture.height;


        uiImage.name.clear();


        return uiImage;
    };


    // =====================================================
    // 8.2 更新 Single Preview
    // =====================================================

    auto updateSinglePreview =
        [&]()
    {
        const ImageTexture* texture =
            processManager.getSingleTexture();


        if (texture == nullptr)
        {
            mainUi.clearSinglePreview();

            return;
        }


        mainUi.setSinglePreview(
            makeUiImage(
                *texture
            )
        );
    };


    // =====================================================
    // 8.3 更新 Batch Preview
    // =====================================================

    auto updateBatchPreview =
        [&]()
    {
        const ImageTexture* texture =
            processManager.getBatchTexture();


        if (texture == nullptr)
        {
            mainUi.clearBatchPreview();

            return;
        }


        mainUi.setBatchPreview(
            makeUiImage(
                *texture
            )
        );
    };


    // =====================================================
    // 8.4 加载某一个 Group 的全部图片
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
        // 加载 Group 图片
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


            uiImage.textureId =
                static_cast<ImTextureID>(
                    image.texture.id
                );


            uiImage.width =
                image.texture.width;


            uiImage.height =
                image.texture.height;


            // 暂时不显示文件名
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


        std::cout
            << "Loaded group "
            << groupIndex + 1
            << ", images: "
            << imageManager.getImageCount()
            << '\n';
    };


    // =====================================================
    // 8.5 选择 Group Root Folder
    // =====================================================

    uiEvents.onSelectGroupFolder =
        [&]()
    {
        groupFolderDialog.open(
            window
        );
    };


    // =====================================================
    // 8.6 重新选择 Group Root Folder
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
    // 8.7 点击顶部 Group
    // =====================================================

    uiEvents.onGroupClick =
        [&](int groupIndex)
    {
        loadGroup(
            groupIndex
        );
    };


    // =====================================================
    // 8.8 点击左侧图片
    //
    // 暂时只保留接口
    // =====================================================

    uiEvents.onImageClick =
        [&](int groupIndex, int imageIndex)
    {
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
        // 左侧图片点击业务
        //
        // 当前可以获得：
        //
        // groupIndex
        // imageIndex
        // image->path
        //
        // ======================================

        (void)groupIndex;
    };


    // =====================================================
    // 8.9 Single - Select Image
    // =====================================================

    uiEvents.onPickSingle =
        [&]()
    {
        singleFileDialog.open(
            window
        );
    };


    // =====================================================
    // 8.10 Single - Drop Image
    // =====================================================

    uiEvents.onSingleDrop =
        [&](const std::string& path)
    {
        if (
            processManager.loadSingle(
                path
            )
        )
        {
            updateSinglePreview();


            std::cout
                << "Single image loaded."
                << '\n';
        }
        else
        {
            std::cerr
                << "Failed to load single image: "
                << processManager.getLastError()
                << '\n';
        }
    };


    // =====================================================
    // 8.11 Single - Confirm
    //
    // 暂时只保留接口
    // =====================================================

    uiEvents.onSingleConfirm =
        [&]()
    {
        // ======================================
        // TODO:
        //
        // Single Confirm
        //
        // ======================================
    };


    // =====================================================
    // 8.12 Batch - Select Folder
    // =====================================================

    uiEvents.onPickFolder =
        [&]()
    {
        batchFolderDialog.open(
            window
        );
    };


    // =====================================================
    // 8.13 Batch - Previous
    // =====================================================

    uiEvents.onBatchPrev =
        [&]()
    {
        if (
            processManager.prevBatch()
        )
        {
            updateBatchPreview();


            std::cout
                << "Batch image: "
                << processManager.getBatchIndex() + 1
                << " / "
                << processManager.getBatchCount()
                << '\n';
        }
    };


    // =====================================================
    // 8.14 Batch - Next
    // =====================================================

    uiEvents.onBatchNext =
        [&]()
    {
        if (
            processManager.nextBatch()
        )
        {
            updateBatchPreview();


            std::cout
                << "Batch image: "
                << processManager.getBatchIndex() + 1
                << " / "
                << processManager.getBatchCount()
                << '\n';
        }
    };


    // =====================================================
    // 8.15 Batch - Confirm
    //
    // 暂时只保留接口
    // =====================================================

    uiEvents.onBatchConfirm =
        [&]()
    {
        // ======================================
        // TODO:
        //
        // Batch Confirm
        //
        // ======================================
    };


    // =====================================================
    // 8.16 New Category
    //
    // 暂时只保留接口
    // =====================================================

    uiEvents.onNewCategory =
        [&]()
    {
        // ======================================
        // TODO:
        //
        // Create New Category
        //
        // ======================================
    };


    // =====================================================
    // 8.17 注册 UI Events
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
            // 先让 ImGui 处理输入
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


            // ---------------------------------------------
            // Single Image Drag & Drop
            //
            // MainUi -> ProcessPanel 会判断
            // 当前是不是 Single 模式。
            // ---------------------------------------------

            if (
                event.type ==
                    SDL_EVENT_DROP_FILE
                &&
                event.drop.data != nullptr
            )
            {
                mainUi.handleFileDrop(
                    std::string(
                        event.drop.data
                    )
                );
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
                // 清理旧 Group 图片
                // -----------------------------------------

                imageManager.clear();

                mainUi.clearCompareImages();


                // -----------------------------------------
                // 自动显示第一个 Group
                // -----------------------------------------

                if (
                    groupManager.getGroupCount()
                    > 0
                )
                {
                    loadGroup(0);
                }


                // -----------------------------------------
                // Debug
                // -----------------------------------------

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
        // 10.3 Group Folder Dialog Error
        // =================================================

        if (
            auto dialogError =
                groupFolderDialog.takeError()
        )
        {
            std::cerr
                << "Group folder dialog error: "
                << *dialogError
                << '\n';
        }


        // =================================================
        // 10.4 Single Image Dialog Result
        // =================================================

        if (
            auto selectedPath =
                singleFileDialog.takePath()
        )
        {
            if (
                processManager.loadSingle(
                    *selectedPath
                )
            )
            {
                updateSinglePreview();


                std::cout
                    << "Single image loaded."
                    << '\n';
            }
            else
            {
                std::cerr
                    << "Failed to load single image: "
                    << processManager.getLastError()
                    << '\n';
            }
        }


        // =================================================
        // 10.5 Single Image Dialog Error
        // =================================================

        if (
            auto dialogError =
                singleFileDialog.takeError()
        )
        {
            std::cerr
                << "Image dialog error: "
                << *dialogError
                << '\n';
        }


        // =================================================
        // 10.6 Batch Folder Dialog Result
        // =================================================

        if (
            auto selectedPath =
                batchFolderDialog.takePath()
        )
        {
            if (
                processManager.loadBatchFolder(
                    *selectedPath
                )
            )
            {
                updateBatchPreview();


                std::cout
                    << "Batch folder loaded."
                    << '\n';


                std::cout
                    << "Batch images: "
                    << processManager.getBatchCount()
                    << '\n';


                std::cout
                    << "Batch image: "
                    << processManager.getBatchIndex() + 1
                    << " / "
                    << processManager.getBatchCount()
                    << '\n';
            }
            else
            {
                std::cerr
                    << "Failed to load batch folder: "
                    << processManager.getLastError()
                    << '\n';
            }
        }


        // =================================================
        // 10.7 Batch Folder Dialog Error
        // =================================================

        if (
            auto dialogError =
                batchFolderDialog.takeError()
        )
        {
            std::cerr
                << "Batch folder dialog error: "
                << *dialogError
                << '\n';
        }


        // =================================================
        // 10.8 窗口最小化
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
        // 10.9 Start ImGui Frame
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


        // ---------------------------------------------
        // Retina / High DPI framebuffer size
        // ---------------------------------------------

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
    //
    // ImageManager 和 ProcessManager
    // 都持有 OpenGL Texture。
    //
    // 必须先释放 Texture，
    // 再销毁 OpenGL Context。
    // =====================================================

    processManager.clear();

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
