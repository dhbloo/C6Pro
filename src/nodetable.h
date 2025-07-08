#pragma once

#include "node.h"

#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>

class NodeTable
{
public:
    struct NodeCompare
    {
        using is_transparent = void;
        bool operator()(uint64_t lhs, const Node &rhs) const { return lhs < rhs.getHash(); }
        bool operator()(const Node &lhs, uint64_t rhs) const { return lhs.getHash() < rhs; }
        bool operator()(const Node &lhs, const Node &rhs) const
        {
            return lhs.getHash() < rhs.getHash();
        }
    };
    using Table = std::set<Node, NodeCompare>;

    struct Shard
    {
        size_t             index;
        Table             &table;
        std::shared_mutex &mutex;
    };

    NodeTable(size_t numShardsPowerOfTwo)
        : numShards_(1 << numShardsPowerOfTwo)
        , mask_(numShards_ - 1)
        , tables_(std::make_unique<Table[]>(numShards_))
        , mutexes_(std::make_unique<std::shared_mutex[]>(numShards_))
    {}

    /// Get the total number of shards of this node table.
    size_t getNumShards() const { return numShards_; }

    /// Get the shard with the given shard index.
    /// @note This function is thread-safe.
    Shard getShardByShardIndex(size_t index) const
    {
        return Shard {index, tables_[index], mutexes_[index]};
    }

    /// Get the shard that contains the node with the given hash key.
    /// @note This function is thread-safe.
    Shard getShardByHash(uint64_t hash) const
    {
        size_t index = hash & mask_;
        return getShardByShardIndex(index);
    }

    /// Find the node with the given hash key.
    /// @return Pointer to the node if found, otherwise nullptr.
    /// @note This function uses reader lock to ensure thread-safety.
    Node *findNode(uint64_t hash) const
    {
        Shard            shard = getShardByHash(hash);
        std::shared_lock lock(shard.mutex);
        Table::iterator  it = shard.table.find(hash);
        if (it == shard.table.end())
            return nullptr;

        // Normally elements in std::set are immutable, but we only use a node's hash
        // to compare nodes, so we can safely cast away constness here.
        return std::addressof(const_cast<Node &>(*it));
    }

    /// Try emplace a new node into the table.
    /// @param hash Hash key of the new node.
    /// @param args Extra arguments to pass to the constructor of the new node.
    /// @return A pair of (Pointer to the inserted node, Whether the node is
    ///   successfully inserted). If there is already a node inserted by other
    ///   threads, the pointer to that node is returned instead.
    template <typename... Args>
    std::pair<Node *, bool> tryEmplaceNode(uint64_t hash, Args... args)
    {
        Shard            shard = getShardByHash(hash);
        std::unique_lock lock(shard.mutex);

        // Try to emplace the node after acquiring the writer lock
        auto [it, inserted] = shard.table.emplace(hash, std::forward<Args>(args)...);
        // We also return whether the node is actually created by us
        return {std::addressof(const_cast<Node &>(*it)), inserted};
    }

private:
    size_t                               numShards_;
    size_t                               mask_;
    std::unique_ptr<Table[]>             tables_;
    std::unique_ptr<std::shared_mutex[]> mutexes_;
};