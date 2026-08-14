#include "tools/FolderDialog.h"


void FolderDialog::open(
    SDL_Window* window,
    const std::string& defaultPath
)
{
    {
        std::lock_guard<std::mutex>
            lock(mutex);


        // 已经有一个 Dialog 打开，
        // 就不再重复打开。
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


    SDL_ShowOpenFolderDialog(
        &FolderDialog::dialogCallback,
        this,
        window,
        start,
        false
    );
}


void SDLCALL
FolderDialog::dialogCallback(
    void* userdata,
    const char* const* fileList,
    int filter
)
{
    (void)filter;


    auto* dialog =
        static_cast<FolderDialog*>(
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


    // ==========================================
    // SDL Error
    // ==========================================

    if (fileList == nullptr)
    {
        dialog->pendingError =
            SDL_GetError();

        return;
    }


    // ==========================================
    // User Cancel
    // ==========================================

    if (fileList[0] == nullptr)
    {
        return;
    }


    // ==========================================
    // Copy Result
    //
    // SDL 的 fileList 不能长期保存，
    // 所以这里立即复制字符串。
    // ==========================================

    dialog->pendingPath =
        std::string(
            fileList[0]
        );
}


std::optional<std::string>
FolderDialog::takePath()
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
FolderDialog::takeError()
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


bool FolderDialog::isOpen() const
{
    std::lock_guard<std::mutex>
        lock(mutex);


    return opened;
}
