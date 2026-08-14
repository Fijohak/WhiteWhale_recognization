#pragma once

#include <filesystem>
#include <string>


struct ImageTexture
{
    unsigned int id = 0;

    int width = 0;

    int height = 0;


    bool valid() const
    {
        return
            id != 0 &&
            width > 0 &&
            height > 0;
    }
};


class ImageLoader
{
public:

    static bool isImageFile(
        const std::filesystem::path& path
    );


    static bool load(
        const std::filesystem::path& path,
        ImageTexture& texture,
        std::string& error
    );


    static void release(
        ImageTexture& texture
    );
};
