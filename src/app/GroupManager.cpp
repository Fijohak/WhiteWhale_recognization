#include "app/GroupManager.h"

#include <algorithm>
#include <system_error>
#include <utility>


namespace fs = std::filesystem;


namespace
{

std::string pathToUtf8(
    const fs::path& path
)
{
#if defined(__cpp_char8_t)

    const auto text =
        path.u8string();

    return std::string(
        reinterpret_cast<const char*>(
            text.data()
        ),
        text.size()
    );

#else

    return path.u8string();

#endif
}


fs::path pathFromUtf8(
    const std::string& path
)
{
    return fs::u8path(path);
}

}


bool GroupManager::loadRoot(
    const std::string& newRootPath
)
{
    lastError.clear();


    if (newRootPath.empty())
    {
        lastError =
            "Group root path is empty.";

        return false;
    }


    fs::path root =
        pathFromUtf8(
            newRootPath
        );


    std::error_code error;


    // ==========================================
    // 检查目录是否存在
    // ==========================================

    if (!fs::exists(root, error))
    {
        lastError =
            "Selected folder does not exist.";

        return false;
    }


    if (error)
    {
        lastError =
            "Failed to access selected folder.";

        return false;
    }


    // ==========================================
    // 检查是不是目录
    // ==========================================

    if (!fs::is_directory(root, error))
    {
        lastError =
            "Selected path is not a folder.";

        return false;
    }


    if (error)
    {
        lastError =
            "Failed to inspect selected folder.";

        return false;
    }


    // 尽量转成绝对路径
    fs::path absoluteRoot =
        fs::absolute(
            root,
            error
        );


    if (!error)
    {
        root =
            absoluteRoot.lexically_normal();
    }


    // ==========================================
    // 临时 Group 数据
    //
    // 只有全部扫描完成后，
    // 才替换当前数据。
    // ==========================================

    std::vector<GroupInfo>
        newGroups;


    try
    {
        // ======================================
        // 只扫描一级子目录
        // ======================================

        for (
            const auto& entry :
            fs::directory_iterator(
                root,
                fs::directory_options::
                    skip_permission_denied
            )
        )
        {
            std::error_code typeError;


            if (
                !entry.is_directory(
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


            GroupInfo group;


            group.name =
                pathToUtf8(
                    entry.path().filename()
                );


            group.path =
                entry.path().lexically_normal();


            newGroups.push_back(
                std::move(group)
            );
        }
    }
    catch (
        const fs::filesystem_error& e
    )
    {
        lastError =
            e.what();

        return false;
    }


    // ==========================================
    // 排序
    //
    // 不依赖 filesystem 返回顺序。
    // ==========================================

    std::sort(
        newGroups.begin(),
        newGroups.end(),
        [](
            const GroupInfo& left,
            const GroupInfo& right
        )
        {
            return left.name
                <
                right.name;
        }
    );


    // ==========================================
    // 全部成功后替换当前状态
    // ==========================================

    rootPath =
        pathToUtf8(root);


    rootName =
        pathToUtf8(
            root.filename()
        );


    // 某些根路径 filename 可能为空
    if (rootName.empty())
    {
        rootName =
            rootPath;
    }


    groups =
        std::move(
            newGroups
        );


    return true;
}


void GroupManager::clear()
{
    rootPath.clear();

    rootName.clear();

    groups.clear();

    lastError.clear();
}


int GroupManager::getGroupCount() const
{
    return static_cast<int>(
        groups.size()
    );
}


const GroupInfo*
GroupManager::getGroup(
    int index
) const
{
    if (
        index < 0
        ||
        index >= getGroupCount()
    )
    {
        return nullptr;
    }


    return &groups[index];
}


const std::vector<GroupInfo>&
GroupManager::getGroups() const
{
    return groups;
}


const std::string&
GroupManager::getRootPath() const
{
    return rootPath;
}


const std::string&
GroupManager::getRootName() const
{
    return rootName;
}


const std::string&
GroupManager::getLastError() const
{
    return lastError;
}
