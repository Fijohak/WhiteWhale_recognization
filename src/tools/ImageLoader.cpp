#include "tools/ImageLoader.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <vector>

#include <glad/gl.h>


#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>


namespace fs = std::filesystem;


bool ImageLoader::isImageFile(
    const fs::path& path
)
{
    if (!path.has_extension())
    {
        return false;
    }


    std::string extension =
        path.extension().string();


    std::transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](unsigned char value)
        {
            return static_cast<char>(
                std::tolower(value)
            );
        }
    );


    return
        extension == ".png" ||
        extension == ".jpg" ||
        extension == ".jpeg" ||
        extension == ".bmp" ||
        extension == ".tga";
}


bool ImageLoader::load(
    const fs::path& path,
    ImageTexture& texture,
    std::string& error
)
{
    error.clear();


    // ==========================================
    // 如果 texture 之前有数据，先释放
    // ==========================================

    release(texture);


    // ==========================================
    // 读取文件到内存
    //
    // 不直接使用 stbi_load(path.c_str())
    // 可以更好地兼容 filesystem path。
    // ==========================================

    std::ifstream file(
        path,
        std::ios::binary |
        std::ios::ate
    );


    if (!file.is_open())
    {
        error =
            "Failed to open image file.";

        return false;
    }


    const std::streamsize fileSize =
        file.tellg();


    if (fileSize <= 0)
    {
        error =
            "Image file is empty.";

        return false;
    }


    if (
        fileSize >
        static_cast<std::streamsize>(
            std::numeric_limits<int>::max()
        )
    )
    {
        error =
            "Image file is too large.";

        return false;
    }


    file.seekg(
        0,
        std::ios::beg
    );


    std::vector<unsigned char> buffer(
        static_cast<std::size_t>(
            fileSize
        )
    );


    if (
        !file.read(
            reinterpret_cast<char*>(
                buffer.data()
            ),
            fileSize
        )
    )
    {
        error =
            "Failed to read image file.";

        return false;
    }


    // ==========================================
    // stb_image 解码
    // ==========================================

    int width = 0;

    int height = 0;

    int channels = 0;


    unsigned char* pixels =
        stbi_load_from_memory(
            buffer.data(),
            static_cast<int>(
                buffer.size()
            ),
            &width,
            &height,
            &channels,
            STBI_rgb_alpha
        );


    if (pixels == nullptr)
    {
        const char* reason =
            stbi_failure_reason();


        error =
            reason != nullptr
                ? reason
                : "Failed to decode image.";

        return false;
    }


    // ==========================================
    // 创建 OpenGL Texture
    // ==========================================

    GLuint textureId = 0;


    GLint oldTexture = 0;

    GLint oldAlignment = 0;


    glGetIntegerv(
        GL_TEXTURE_BINDING_2D,
        &oldTexture
    );


    glGetIntegerv(
        GL_UNPACK_ALIGNMENT,
        &oldAlignment
    );


    glGenTextures(
        1,
        &textureId
    );


    if (textureId == 0)
    {
        stbi_image_free(
            pixels
        );


        error =
            "Failed to create OpenGL texture.";

        return false;
    }


    glBindTexture(
        GL_TEXTURE_2D,
        textureId
    );


    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MIN_FILTER,
        GL_LINEAR
    );


    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_LINEAR
    );


    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_WRAP_S,
        GL_CLAMP_TO_EDGE
    );


    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_WRAP_T,
        GL_CLAMP_TO_EDGE
    );


    glPixelStorei(
        GL_UNPACK_ALIGNMENT,
        1
    );


    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA8,
        width,
        height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        pixels
    );


    // ==========================================
    // 恢复 OpenGL 状态
    // ==========================================

    glPixelStorei(
        GL_UNPACK_ALIGNMENT,
        oldAlignment
    );


    glBindTexture(
        GL_TEXTURE_2D,
        static_cast<GLuint>(
            oldTexture
        )
    );


    // CPU 图片数据已经不需要
    stbi_image_free(
        pixels
    );


    // ==========================================
    // 输出 Texture
    // ==========================================

    texture.id =
        textureId;


    texture.width =
        width;


    texture.height =
        height;


    return true;
}


void ImageLoader::release(
    ImageTexture& texture
)
{
    if (texture.id != 0)
    {
        const GLuint textureId =
            texture.id;


        glDeleteTextures(
            1,
            &textureId
        );
    }


    texture.id = 0;

    texture.width = 0;

    texture.height = 0;
}
