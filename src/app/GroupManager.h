#pragma once

#include <filesystem>
#include <string>
#include <vector>


struct GroupInfo
{
    std::string name;

    std::filesystem::path path;
};


class GroupManager
{
public:

    // ==========================================
    // Root
    // ==========================================

    bool loadRoot(
        const std::string& rootPath
    );


    void clear();


    // ==========================================
    // Group
    // ==========================================

    int getGroupCount() const;


    const GroupInfo* getGroup(
        int index
    ) const;


    const std::vector<GroupInfo>&
    getGroups() const;


    // ==========================================
    // Image Classification
    // ==========================================

    bool copyImageToGroup(
        int groupIndex,
        const std::filesystem::path& imagePath
    );


    bool createGroupWithImage(
        const std::filesystem::path& imagePath,
        int& newGroupIndex
    );


    // ==========================================
    // Root Info
    // ==========================================

    const std::string&
    getRootPath() const;


    const std::string&
    getRootName() const;


    const std::string&
    getLastError() const;


private:

    int findGroupIndex(
        const std::filesystem::path& path
    ) const;


private:

    std::string rootPath;

    std::string rootName;

    std::vector<GroupInfo> groups;

    std::string lastError;
};
