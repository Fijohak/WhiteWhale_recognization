cmake_minimum_required(VERSION 3.20)

project(ImageViewer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# ==========================================
# SDL3
# ==========================================

add_subdirectory(
    external/SDL
    EXCLUDE_FROM_ALL
)


# ==========================================
# OpenGL
# ==========================================

find_package(OpenGL REQUIRED)


# ==========================================
# Executable
# ==========================================

add_executable(ImageViewer

    # ------------------------------
    # 我们自己的代码
    # ------------------------------

    src/main.cpp
    src/main.h
    src/ui/TopBar.cpp
    src/ui/TopBar.h

    src/ui/ComparePanel.cpp
    src/ui/ComparePanel.h

    src/ui/ProcessPanel.cpp
    src/ui/ProcessPanel.h

    src/ui/MainUi.cpp
    src/ui/MainUi.h

    src/ui/UiImage.h

    src/app/GroupManager.cpp
    src/app/GroupManager.h

    src/app/ImageManager.cpp
    src/app/ImageManager.h

    src/app/ProcessManager.cpp
    src/app/ProcessManager.h

    src/tools/FolderDialog.cpp
    src/tools/FolderDialog.h

    src/tools/ImageLoader.cpp
    src/tools/ImageLoader.h

    src/tools/FileDialog.cpp
    src/tools/FileDialog.h


    # ------------------------------
    # GLAD
    # ------------------------------

    external/glad/src/gl.c


    # ------------------------------
    # Dear ImGui
    # ------------------------------

    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/imgui_demo.cpp


    # ------------------------------
    # ImGui Backend
    # ------------------------------

    external/imgui/backends/imgui_impl_sdl3.cpp
    external/imgui/backends/imgui_impl_opengl3.cpp
)


# ==========================================
# Include
# ==========================================

target_include_directories(ImageViewer
    PRIVATE

    src

    external/glad/include

    external/imgui
    external/imgui/backends

    external/stb
)


# ==========================================
# Link
# ==========================================

target_link_libraries(ImageViewer
    PRIVATE

    SDL3::SDL3
    OpenGL::GL
)
