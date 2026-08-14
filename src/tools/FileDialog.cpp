#include "tools/FileDialog.h"


void FileDialog::open(
    SDL_Window* window,
    const std::string& defaultPath
)
{
    {
        std::lock_guard<std::mutex>
            lock(mutex);


        if (opened)
        {
            return;
        }


        opened = true;

        pendingPath.reset();

        pendingError.reset();

        startPath =
            defaultPath;
    }


    const char* start =
        startPath.empty()
            ? nullptr
            : startPath.c_str();


    // 必须保持到异步 callback 执行为止。
    // static 生命周期满足要求。
    static const SDL_DialogFileFilter
        filters[] =
    {
        {
            "Images",
            "png;jpg;jpeg;bmp;tga"
        }
    };


    SDL_ShowOpenFileDialog(
        &FileDialog::dialogCallback,
        this,
        window,
        filters,
        1,
        start,
        false
    );
}


void SDLCALL
FileDialog::dialogCallback(
    void* userdata,
    const char* const* fileList,
    int filter
)
{
    (void)filter;


    auto* dialog =
        static_cast<FileDialog*>(
            userdata
        );


    if (dialog == nullptr)
    {
        return;
    }


    std::lock_guard<std::mutex>
        lock(dialog->mutex);


    dialog->opened =
        false;


    // SDL Error
    if (fileList == nullptr)
    {
        dialog->pendingError =
            SDL_GetError();

        return;
    }


    // User Cancel
    if (fileList[0] == nullptr)
    {
        return;
    }


    // SDL 的 fileList 只能在 callback
    // 内有效，所以立即复制。
    dialog->pendingPath =
        std::string(
            fileList[0]
        );
}


std::optional<std::string>
FileDialog::takePath()
{
    std::lock_guard<std::mutex>
        lock(mutex);


    if (!pendingPath)
    {
        return std::nullopt;
    }


    auto result =
        std::move(
            pendingPath
        );


    pendingPath.reset();


    return result;
}


std::optional<std::string>
FileDialog::takeError()
{
    std::lock_guard<std::mutex>
        lock(mutex);


    if (!pendingError)
    {
        return std::nullopt;
    }


    auto result =
        std::move(
            pendingError
        );


    pendingError.reset();


    return result;
}


bool FileDialog::isOpen() const
{
    std::lock_guard<std::mutex>
        lock(mutex);


    return opened;
}
