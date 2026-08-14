#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "tools/ImageLoader.h"


class ProcessManager
{
public:

    ProcessManager() = default;


    ~ProcessManager();


    ProcessManager(
        const ProcessManager&
    ) = delete;


    ProcessManager& operator=(
        const ProcessManager&
    ) = delete;


    // ==========================================
    // Single
    // ==========================================

    bool loadSingle(
        const std::string& path
    );


    void clearSingle();


    const ImageTexture*
    getSingleTexture() const;


    const std::filesystem::path&
    getSinglePath() const;


    // ==========================================
    // Batch
    // ==========================================

    bool loadBatchFolder(
        const std::string& folderPath
    );


    bool prevBatch();


    bool nextBatch();


    void clearBatch();


    const ImageTexture*
    getBatchTexture() const;


    const std::filesystem::path&
    getBatchPath() const;


    int getBatchIndex() const;


    int getBatchCount() const;


    // ==========================================
    // All
    // ==========================================

    void clear();


    const std::string&
    getLastError() const;


private:

    bool loadBatchIndex(
        std::size_t index
    );


private:

    // ==========================================
    // Single
    // ==========================================

    std::filesystem::path
        singlePath;


    ImageTexture
        singleTexture;


    // ==========================================
    // Batch
    // ==========================================

    std::filesystem::path
        batchFolder;


    std::vector<std::filesystem::path>
        batchPaths;


    std::size_t batchIndex = 0;


    std::filesystem::path
        batchPath;


    ImageTexture
        batchTexture;


    // ==========================================
    // Error
    // ==========================================

    std::string lastError;
};
