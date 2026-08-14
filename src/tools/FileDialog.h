#pragma once

#include <mutex>
#include <optional>
#include <string>

#include <SDL3/SDL.h>
#include <SDL3/SDL_dialog.h>


class FileDialog
{
public:

    void open(
        SDL_Window* window,
        const std::string& defaultPath = {}
    );


    std::optional<std::string>
    takePath();


    std::optional<std::string>
    takeError();


    bool isOpen() const;


private:

    static void SDLCALL
    dialogCallback(
        void* userdata,
        const char* const* fileList,
        int filter
    );


private:

    mutable std::mutex mutex;

    bool opened = false;

    std::string startPath;

    std::optional<std::string>
        pendingPath;

    std::optional<std::string>
        pendingError;
};
