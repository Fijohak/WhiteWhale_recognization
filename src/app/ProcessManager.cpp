#include "app/ProcessManager.h"

#include <algorithm>
#include <system_error>
#include <utility>


namespace fs = std::filesystem;


namespace
{

fs::path pathFromUtf8(
    const std::string& path
)
{
    return fs::u8path(
        path
    );
}

}


ProcessManager::~ProcessManager()
{
    clear();
}


// =========================================================
// Single Image
// =========================================================

bool ProcessManager::loadSingle(
    const std::string& path
)
{
    lastError.clear();


    if (path.empty())
    {
        lastError =
            "Image path is empty.";

        return false;
    }


    const fs::path imagePath =
        pathFromUtf8(
            path
        );


    std::error_code error;


    if (
        !fs::exists(
            imagePath,
            error
        )
    )
    {
        lastError =
            "Image file does not exist.";

        return false;
    }


    if (
        !fs::is_regular_file(
            imagePath,
            error
        )
    )
    {
        lastError =
            "Selected path is not a file.";

        return false;
    }


    if (
        !ImageLoader::isImageFile(
            imagePath
        )
    )
    {
        lastError =
            "Selected file is not a supported image.";

        return false;
    }


    // ==========================================
    // 先加载新图。
    //
    // 成功以后再删除旧图，避免用户选到
    // 损坏图片时旧预览突然消失。
    // ==========================================

    ImageTexture
        newTexture;


    std::string
        loadError;


    if (
        !ImageLoader::load(
            imagePath,
            newTexture,
            loadError
        )
    )
    {
        lastError =
            loadError;

        return false;
    }


    // ==========================================
    // Replace
    // ==========================================

    ImageLoader::release(
        singleTexture
    );


    singleTexture =
        newTexture;


    singlePath =
        imagePath;


    return true;
}


// =========================================================
// Clear Single
// =========================================================

void ProcessManager::clearSingle()
{
    ImageLoader::release(
        singleTexture
    );


    singlePath.clear();
}


// =========================================================
// Single Texture
// =========================================================

const ImageTexture*
ProcessManager::getSingleTexture() const
{
    if (!singleTexture.valid())
    {
        return nullptr;
    }


    return &singleTexture;
}


// =========================================================
// Single Path
// =========================================================

const fs::path&
ProcessManager::getSinglePath() const
{
    return singlePath;
}


// =========================================================
// Load Batch Folder
// =========================================================

bool ProcessManager::loadBatchFolder(
    const std::string& folderPath
)
{
    lastError.clear();


    if (folderPath.empty())
    {
        lastError =
            "Batch folder path is empty.";

        return false;
    }


    const fs::path folder =
        pathFromUtf8(
            folderPath
        );


    std::error_code error;


    if (
        !fs::exists(
            folder,
            error
        )
    )
    {
        lastError =
            "Batch folder does not exist.";

        return false;
    }


    if (
        !fs::is_directory(
            folder,
            error
        )
    )
    {
        lastError =
            "Batch path is not a folder.";

        return false;
    }


    // ==========================================
    // 扫描一级图片
    // ==========================================

    std::vector<fs::path>
        newPaths;


    try
    {
        for (
            const auto& entry :
            fs::directory_iterator(
                folder,
                fs::directory_options::
                    skip_permission_denied
            )
        )
        {
            std::error_code
                typeError;


            if (
                !entry.is_regular_file(
                    typeError
                )
            )
            {
                continue;
            }


            if (typeError)
            {
                continue;
            }


            if (
                !ImageLoader::isImageFile(
                    entry.path()
                )
            )
            {
                continue;
            }


            newPaths.push_back(
                entry.path()
            );
        }
    }
    catch (
        const fs::filesystem_error& exception
    )
    {
        lastError =
            exception.what();

        return false;
    }


    // ==========================================
    // 排序
    //
    // 保证 Prev / Next 顺序稳定。
    // ==========================================

    std::sort(
        newPaths.begin(),
        newPaths.end()
    );


    if (newPaths.empty())
    {
        lastError =
            "No images found in selected folder.";

        return false;
    }


    // ==========================================
    // 找第一张能够成功读取的图片
    // ==========================================

    ImageTexture
        firstTexture;


    std::size_t firstIndex = 0;


    bool found = false;


    std::string loadError;


    for (
        std::size_t i = 0;
        i < newPaths.size();
        ++i
    )
    {
        if (
            ImageLoader::load(
                newPaths[i],
                firstTexture,
                loadError
            )
        )
        {
            firstIndex = i;

            found = true;

            break;
        }
    }


    if (!found)
    {
        lastError =
            "No readable images found in selected folder.";

        return false;
    }


    // ==========================================
    // 新目录成功以后才删除旧 Batch
    // ==========================================

    ImageLoader::release(
        batchTexture
    );


    batchFolder =
        folder;


    batchPaths =
        std::move(
            newPaths
        );


    batchIndex =
        firstIndex;


    batchPath =
        batchPaths[
            batchIndex
        ];


    batchTexture =
        firstTexture;


    return true;
}


// =========================================================
// Load Batch Index
// =========================================================

bool ProcessManager::loadBatchIndex(
    std::size_t index
)
{
    if (
        index >=
        batchPaths.size()
    )
    {
        return false;
    }


    ImageTexture
        newTexture;


    std::string
        loadError;


    if (
        !ImageLoader::load(
            batchPaths[index],
            newTexture,
            loadError
        )
    )
    {
        lastError =
            loadError;

        return false;
    }


    // 新图已经加载成功，
    // 再释放旧 Texture。
    ImageLoader::release(
        batchTexture
    );


    batchTexture =
        newTexture;


    batchIndex =
        index;


    batchPath =
        batchPaths[index];


    return true;
}


// =========================================================
// Previous
// =========================================================

bool ProcessManager::prevBatch()
{
    if (batchPaths.empty())
    {
        return false;
    }


    // 已经是第一张。
    //
    // 什么都不做。
    if (batchIndex == 0)
    {
        return false;
    }


    // 如果中间存在损坏图片，
    // 自动继续向前找。
    std::size_t index =
        batchIndex;


    while (index > 0)
    {
        --index;


        if (
            loadBatchIndex(
                index
            )
        )
        {
            return true;
        }
    }


    return false;
}


// =========================================================
// Next
// =========================================================

bool ProcessManager::nextBatch()
{
    if (batchPaths.empty())
    {
        return false;
    }


    // 已经是最后一张。
    //
    // 什么都不做。
    if (
        batchIndex + 1
        >=
        batchPaths.size()
    )
    {
        return false;
    }


    // 如果有损坏图片，
    // 自动继续向后找。
    for (
        std::size_t index =
            batchIndex + 1;

        index <
        batchPaths.size();

        ++index
    )
    {
        if (
            loadBatchIndex(
                index
            )
        )
        {
            return true;
        }
    }


    return false;
}


// =========================================================
// Clear Batch
// =========================================================

void ProcessManager::clearBatch()
{
    ImageLoader::release(
        batchTexture
    );


    batchFolder.clear();

    batchPaths.clear();

    batchPath.clear();

    batchIndex = 0;
}


// =========================================================
// Batch Texture
// =========================================================

const ImageTexture*
ProcessManager::getBatchTexture() const
{
    if (!batchTexture.valid())
    {
        return nullptr;
    }


    return &batchTexture;
}


// =========================================================
// Batch Path
// =========================================================

const fs::path&
ProcessManager::getBatchPath() const
{
    return batchPath;
}


// =========================================================
// Batch Index
// =========================================================

int ProcessManager::getBatchIndex() const
{
    if (batchPaths.empty())
    {
        return -1;
    }


    return static_cast<int>(
        batchIndex
    );
}


// =========================================================
// Batch Count
// =========================================================

int ProcessManager::getBatchCount() const
{
    return static_cast<int>(
        batchPaths.size()
    );
}


// =========================================================
// Clear All
// =========================================================

void ProcessManager::clear()
{
    clearSingle();

    clearBatch();

    lastError.clear();
}


// =========================================================
// Error
// =========================================================

const std::string&
ProcessManager::getLastError() const
{
    return lastError;
}
