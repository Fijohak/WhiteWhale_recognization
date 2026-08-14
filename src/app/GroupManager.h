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

    bool loadRoot(
        const std::string& rootPath
    );


    void clear();


    int getGroupCount() const;


    const GroupInfo* getGroup(
        int index
    ) const;


    const std::vector<GroupInfo>&
    getGroups() const;


    const std::string&
    getRootPath() const;


    const std::string&
    getRootName() const;


    const std::string&
    getLastError() const;


private:

    std::string rootPath;

    std::string rootName;

    std::vector<GroupInfo> groups;

    std::string lastError;
};
