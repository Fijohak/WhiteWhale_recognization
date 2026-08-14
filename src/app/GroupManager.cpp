#include "app/GroupManager.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <system_error>
#include <utility>

#include "tools/FileUtils.h"


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
    return fs::u8path(
        path
    );
}


std::string makeGroupName(
    int number
)
{
    std::ostringstream stream;


    stream
        << "group_"
        << std::setw(4)
        << std::setfill('0')
        << number;


    return stream.str();
}

}


// =========================================================
// Load Root
// =========================================================

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
    // Exists
    // ==========================================

    if (
        !fs::exists(
            root,
            error
        )
    )
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
    // Directory
    // ==========================================

    if (
        !fs::is_directory(
            root,
            error
        )
    )
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


    // ==========================================
    // Absolute Path
    // ==========================================

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
    // Scan Groups
    // ==========================================

    std::vector<GroupInfo>
        newGroups;


    try
    {
        for (
            const auto& entry :
            fs::directory_iterator(
                root,
                fs::directory_options::
                    skip_permission_denied
            )
        )
        {
            std::error_code
                typeError;


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
                    entry
                        .path()
                        .filename()
                );


            group.path =
                entry
                    .path()
                    .lexically_normal();


            newGroups.push_back(
                std::move(
                    group
                )
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
    // Stable Order
    // ==========================================

    std::sort(
        newGroups.begin(),
        newGroups.end(),
        [](
            const GroupInfo& left,
            const GroupInfo& right
        )
        {
            return
                left.name <
                right.name;
        }
    );


    // ==========================================
    // Commit
    // ==========================================

    rootPath =
        pathToUtf8(
            root
        );


    rootName =
        pathToUtf8(
            root.filename()
        );


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


// =========================================================
// Clear
// =========================================================

void GroupManager::clear()
{
    rootPath.clear();

    rootName.clear();

    groups.clear();

    lastError.clear();
}


// =========================================================
// Group Count
// =========================================================

int GroupManager::getGroupCount() const
{
    return static_cast<int>(
        groups.size()
    );
}


// =========================================================
// Get Group
// =========================================================

const GroupInfo*
GroupManager::getGroup(
    int index
) const
{
    if (
        index < 0
        ||
        index >=
            getGroupCount()
    )
    {
        return nullptr;
    }


    return &groups[index];
}


// =========================================================
// Groups
// =========================================================

const std::vector<GroupInfo>&
GroupManager::getGroups() const
{
    return groups;
}


// =========================================================
// Copy Image To Existing Group
// =========================================================

bool GroupManager::copyImageToGroup(
    int groupIndex,
    const fs::path& imagePath
)
{
    lastError.clear();


    // 没有选择 Root
    if (rootPath.empty())
    {
        return false;
    }


    const GroupInfo* group =
        getGroup(
            groupIndex
        );


    if (group == nullptr)
    {
        return false;
    }


    fs::path copiedPath;

    std::string copyError;


    if (
        !FileUtils::copyFileUnique(
            imagePath,
            group->path,
            copiedPath,
            copyError
        )
    )
    {
        lastError =
            copyError;

        return false;
    }


    return true;
}


// =========================================================
// Create New Group + Copy Image
// =========================================================

bool GroupManager::createGroupWithImage(
    const fs::path& imagePath,
    int& newGroupIndex
)
{
    newGroupIndex = -1;

    lastError.clear();


    // ==========================================
    // 没有选择 Root
    // ==========================================

    if (rootPath.empty())
    {
        return false;
    }


    const fs::path root =
        pathFromUtf8(
            rootPath
        );


    std::error_code error;


    // ==========================================
    // 创建唯一 Group Folder
    //
    // group_0001
    // group_0002
    // ...
    // ==========================================

    fs::path newGroupPath;


    int number = 1;


    while (true)
    {
        const std::string name =
            makeGroupName(
                number
            );


        const fs::path candidate =
            root /
            fs::path(name);


        error.clear();


        if (
            !fs::exists(
                candidate,
                error
            )
        )
        {
            newGroupPath =
                candidate;

            break;
        }


        ++number;
    }


    // ==========================================
    // Create Folder
    // ==========================================

    error.clear();


    if (
        !fs::create_directory(
            newGroupPath,
            error
        )
    )
    {
        lastError =
            error
                ? error.message()
                : "Failed to create group folder.";

        return false;
    }


    // ==========================================
    // Copy Image
    // ==========================================

    fs::path copiedPath;

    std::string copyError;


    if (
        !FileUtils::copyFileUnique(
            imagePath,
            newGroupPath,
            copiedPath,
            copyError
        )
    )
    {
        // 图片复制失败，
        // 尽量把刚才创建的空目录删除。
        std::error_code removeError;


        fs::remove(
            newGroupPath,
            removeError
        );


        lastError =
            copyError;

        return false;
    }


    // ==========================================
    // Reload Group Root
    //
    // 让 groups[] 立刻包含新 Group。
    // ==========================================

    const std::string currentRoot =
        rootPath;


    if (
        !loadRoot(
            currentRoot
        )
    )
    {
        return false;
    }


    // ==========================================
    // 找到新 Group 新的 index
    // ==========================================

    newGroupIndex =
        findGroupIndex(
            newGroupPath
        );


    if (newGroupIndex < 0)
    {
        lastError =
            "New group was created but not found after reload.";

        return false;
    }


    return true;
}


// =========================================================
// Find Group
// =========================================================

int GroupManager::findGroupIndex(
    const fs::path& path
) const
{
    const fs::path target =
        path.lexically_normal();


    for (
        int i = 0;
        i < getGroupCount();
        ++i
    )
    {
        if (
            groups[i]
                .path
                .lexically_normal()
            ==
            target
        )
        {
            return i;
        }
    }


    return -1;
}


// =========================================================
// Root Path
// =========================================================

const std::string&
GroupManager::getRootPath() const
{
    return rootPath;
}


// =========================================================
// Root Name
// =========================================================

const std::string&
GroupManager::getRootName() const
{
    return rootName;
}


// =========================================================
// Last Error
// =========================================================

const std::string&
GroupManager::getLastError() const
{
    return lastError;
}
