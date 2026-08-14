#pragma once

#include <string>

#include "imgui.h"


struct UiImage
{
    ImTextureID textureId{};

    int width = 0;

    int height = 0;

    std::string name;


    bool valid() const
    {
        return
            textureId != ImTextureID{} &&
            width > 0 &&
            height > 0;
    }
};
