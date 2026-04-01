#ifndef PSGN_HPP
#define PSGN_HPP

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <random>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <queue>
#include <set>
#include <map>

namespace psgn {

/* ── Constants ───────────────────────────────────────────────────── */
constexpr int MAX_TOKEN_LEN = 64;
constexpr int MAX_NODES = 8192;
constexpr int HASHMAP_CAP = 4096;
constexpr int MAX_EXEMPLARS = 32;
constexpr int MAX_SEQ_BUF = 256;

/* ── SDR Operations ──────────────────────────────────────────────── */
class SDR {
public:
    static std::vector<uint8_t> create(int size) {
        return std::vector<uint8_t>(size, 0);
    }
    
    static void copy(std::vector<uint8_t>& dst, const std::vector<uint8_t>& src) {
        dst = src;
    }
    
    static void clear(std::vector<uint8_t>& s) {
        std::fill(s.begin(), s.end(), 0);
    }
    
    static void OR(std::vector<uint8_t>& dst, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
        for (size_t i = 0; i < dst.size(); ++i) dst[i] = a[i] | b[i];
    }
    
    static void OR_inplace(std::vector<uint8_t>& dst, const std::vector<uint8_t>& src) {
        for (size_t i = 0; i < dst.size(); ++i) dst[i] |= src[i];
    }
    
    static void XOR_shifted_inplace(std::vector<uint8_t>& dst, const std::vector<uint8_t>& src, int shift) {
        int mask = static_cast<int>(dst.size()) - 1;
        int s = shift & mask;
        for (size_t i = 0; i < dst.size(); ++i)
            dst[i] ^= src[(i - s + dst.size()) & mask];
    }
    
    static void OR_shifted_inplace(std::vector<uint8_t>& dst, const std::vector<uint8_t>& src, int shift) {
        int mask = static_cast<int>(dst.size()) - 1;
        int s = shift & mask;
        for (size_t i = 0; i < dst.size(); ++i)
            dst[i] |= src[(i - s + dst.size()) & mask];
    }
    
    static int count(const std::vector<uint8_t>& s) {
        int c = 0;
        for (uint8_t v : s) {
            while (v) { c += v & 1; v >>= 1; }
        }
        return c;
    }
    
    static void random_sample(int n, int k, std::vector<int>& out, std::mt19937& rng) {
        std::vector<int> pool(n);
        for (int i = 0; i < n; ++i) pool[i] = i;
        for (int i = 0; i < k; ++i) {
            std::uniform_int_distribution<> dist(i, n - 1);
            int j = dist(rng);
            std::swap(pool[i], pool[j]);
            out[i] = pool[i];
        }
    }
};

/* ── Tokenizer ───────────────────────────────────────────────────── */
inline std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    const char* p = text.c_str();
    while (*p) {
        if (std::isspace(*p) || *p == ',' || *p == ';' || *p == '(' || *p == ')' ||
            *p == '[' || *p == ']' || *p == '{' || *p == '}' || *p == ':' || *p == '!' ||
            *p == '?' || *p == '"' || *p == '\'') { p++; continue; }
        if (*p == '=' || *p == '+' || *p == '-' || *p == '*' || *p == '/' || *p == '.') {
            tokens.emplace_back(1, *p);
            p++;
            continue;
        }
        if (std::isalnum(*p) || *p == '_' || *p == ':') {
            const char* s = p;
            while (*p && (std::isalnum(*p) || *p == '_' || *p == ':')) p++;
            tokens.emplace_back(s, p - s);
            continue;
        }
        p++;
    }
    return tokens;
}

/* ── SDREncoder ──────────────────────────────────────────────────── */
class SDREncoder {
public:
    int sdr_size;
    double sparsity;
    int num_active_bits;
    std::unordered_map<std::string, int> token_to_id;
    std::vector<std::string> id_to_token;
    std::vector<std::vector<uint8_t>> token_sdrs;
    int next_node_id;
    std::mt19937 rng;
    
    SDREncoder(int size = 1024, double spar = 0.02) 
        : sdr_size(size), sparsity(spar), next_node_id(0), rng(42) {
        num_active_bits = static_cast<int>(sdr_size * sparsity);
        id_to_token.reserve(MAX_NODES);
        token_sdrs.reserve(MAX_NODES);
    }
    
    int get_node_id(const std::string& token) {
        auto it = token_to_id.find(token);
        if (it != token_to_id.end()) return it->second;
        
        if (next_node_id >= MAX_NODES) return -1;
        
        int id = next_node_id++;
        std::vector<uint8_t> sdr = SDR::create(sdr_size);
        std::vector<int> indices(sdr_size);
        SDR::random_sample(sdr_size, num_active_bits, indices, rng);
        for (int i = 0; i < num_active_bits; ++i) sdr[indices[i]] = 1;
        
        token_to_id[token] = id;
        if (static_cast<int>(id_to_token.size()) <= id) {
            id_to_token.resize(id + 1);
            token_sdrs.resize(id + 1);
        }
        id_to_token[id] = token;
        token_sdrs[id] = sdr;
        return id;
    }
    
    const std::vector<uint8_t>& encode(const std::string& token) {
        get_node_id(token);
        return token_sdrs[token_to_id[token]];
    }
    
    const std::string& get_token(int id) const {
        return id_to_token[id];
    }
};

/* ── Edge and Graph Structures ───────────────────────────────────── */
struct Edge {
    int target;
    float weight;
    std::vector<uint8_t> context_sdr;
    std::vector<std::vector<int>> scr_exemplars;
    int num_exemplars;
    Edge* next;
    
    Edge() : target(-1), weight(0), num_exemplars(0), next(nullptr) {}
};

struct Prediction {
    int node_id;
    float score;
    bool operator<(const Prediction& other) const {
        return score > other.score; // Descending order
    }
};

class PredictiveGraph {
public:
    std::vector<Edge*> adjacency;
    std::vector<bool> node_exists;
    int ctx_sdr_size;
    int scr_sdr_size;
    
    PredictiveGraph(int ctx_sz = 0, int scr_sz = 0) 
        : adjacency(MAX_NODES, nullptr), node_exists(MAX_NODES, false),
          ctx_sdr_size(ctx_sz), scr_sdr_size(scr_sz) {}
    
    ~PredictiveGraph() {
        for (auto* e : adjacency) {
            while (e) {
                Edge* nx = e->next;
                delete e;
                e = nx;
            }
        }
    }
    
    void add_node(int id) {
        if (id >= 0 && id < MAX_NODES) node_exists[id] = true;
    }
    
    Edge* find_edge(int from, int to) {
        Edge* e = adjacency[from];
        while (e) { if (e->target == to) return e; e = e->next; }
        return nullptr;
    }
    
    float get_edge_weight(int from, int to) {
        Edge* e = find_edge(from, to);
        return e ? e->weight : 0.0f;
    }
    
    void set_edge_weight(int from, int to, float w, 
                         const std::vector<uint8_t>* ctx, const std::vector<uint8_t>* scr) {
        add_node(from);
        add_node(to);
        
        Edge* e = find_edge(from, to);
        if (!e) {
            e = new Edge();
            e->target = to;
            e->next = adjacency[from];
            adjacency[from] = e;
        }
        e->weight = w;
        
        if (ctx && ctx_sdr_size > 0) {
            if (e->context_sdr.empty()) {
                e->context_sdr = *ctx;
            } else {
                SDR::OR_inplace(e->context_sdr, *ctx);
            }
        }
        
        if (scr && scr_sdr_size > 0 && e->num_exemplars < MAX_EXEMPLARS) {
            int cnt = SDR::count(*scr);
            if (cnt > 0) {
                std::vector<int> indices;
                for (int i = 0; i < scr_sdr_size; ++i) {
                    if ((*scr)[i]) indices.push_back(i);
                }
                if (static_cast<int>(e->scr_exemplars.size()) <= e->num_exemplars) {
                    e->scr_exemplars.resize(e->num_exemplars + 1);
                }
                e->scr_exemplars[e->num_exemplars++] = indices;
            }
        }
    }
    
    std::vector<Prediction> get_predictions(int cur, const std::vector<uint8_t>* scr_bias) {
        std::vector<Prediction> preds;
        if (cur < 0 || cur >= MAX_NODES || !node_exists[cur]) return preds;
        
        for (Edge* e = adjacency[cur]; e; e = e->next) {
            float score = e->weight;
            
            if (scr_bias && e->num_exemplars > 0) {
                float max_evidence = 0.0f;
                for (int ex = 0; ex < e->num_exemplars; ++ex) {
                    int overlap = 0;
                    for (int idx : e->scr_exemplars[ex]) {
                        if ((*scr_bias)[idx]) overlap++;
                    }
                    int count = static_cast<int>(e->scr_exemplars[ex].size());
                    int current_active = SDR::count(*scr_bias);
                    float jaccard = static_cast<float>(overlap) / (count + current_active - overlap);
                    
                    if (jaccard > 0.85f) {
                        float evidence = std::pow(jaccard, 8.0f);
                        if (evidence > max_evidence) max_evidence = evidence;
                    }
                }
                score += max_evidence * 200000.0f;
            }
            
            preds.push_back({e->target, score});
        }
        std::sort(preds.begin(), preds.end());
        return preds;
    }
};

/* ── WorkingMemory ───────────────────────────────────────────────── */
class WorkingMemory {
public:
    int sdr_size;
    std::vector<uint8_t> buffer_sdr;
    std::vector<uint8_t> seed_sdr;
    int held_count;
    std::mt19937 rng;
    
    WorkingMemory(int size = 1024) : sdr_size(size), held_count(0), rng(42) {
        buffer_sdr = SDR::create(sdr_size);
        seed_sdr = SDR::create(sdr_size);
        for (int i = 0; i < sdr_size; ++i) {
            seed_sdr[i] = rng() % 2;
        }
        SDR::copy(buffer_sdr, seed_sdr);
    }
    
    void hold(const std::vector<uint8_t>& sdr) {
        SDR::OR_shifted_inplace(buffer_sdr, sdr, (held_count + 1) * 10007);
        held_count++;
    }
    
    const std::vector<uint8_t>* read() const {
        return held_count > 0 ? &buffer_sdr : nullptr;
    }
    
    void clear() {
        SDR::clear(buffer_sdr);
        held_count = 0;
    }
};

/* ── Maze Solver using PSGN ──────────────────────────────────────── */
class MazeSolver {
private:
    SDREncoder encoder;
    PredictiveGraph graph;
    WorkingMemory wm;
    int rows, cols;
    std::vector<std::string> maze;
    std::pair<int,int> start, end;
    
    struct State {
        int r, c, dir;
        std::string path;
    };
    
public:
    MazeSolver(int r = 10, int c = 10) 
        : encoder(512, 0.05), graph(256, 128), wm(512), rows(r), cols(c) {
        generate_maze();
    }
    
    void generate_maze() {
        maze.assign(rows, std::string(cols, '#'));
        std::mt19937 rng(12345);
        
        // Simple DFS maze generation
        std::vector<std::pair<int,int>> stack;
        std::set<std::pair<int,int>> visited;
        
        int sr = 1, sc = 1;
        maze[sr][sc] = ' ';
        stack.push_back({sr, sc});
        visited.insert({sr, sc});
        
        int dr[] = {-2, 2, 0, 0};
        int dc[] = {0, 0, -2, 2};
        
        while (!stack.empty()) {
            auto [r, c] = stack.back();
            stack.pop_back();
            
            std::vector<int> dirs = {0, 1, 2, 3};
            std::shuffle(dirs.begin(), dirs.end(), rng);
            
            for (int d : dirs) {
                int nr = r + dr[d], nc = c + dc[d];
                if (nr > 0 && nr < rows-1 && nc > 0 && nc < cols-1 && !visited.count({nr, nc})) {
                    maze[nr][nc] = ' ';
                    maze[r + dr[d]/2][c + dc[d]/2] = ' ';
                    visited.insert({nr, nc});
                    stack.push_back({r, c});
                    stack.push_back({nr, nc});
                }
            }
        }
        
        start = {1, 1};
        end = {rows-2, cols-2};
        maze[end.first][end.second] = 'E';
        maze[start.first][start.second] = 'S';
    }
    
    void train_on_solution() {
        // BFS to find solution
        std::queue<State> q;
        std::set<std::pair<int,int>> visited;
        
        q.push({start.first, start.second, 0, ""});
        visited.insert(start);
        
        int dr[] = {-1, 0, 1, 0}; // N, E, S, W
        int dc[] = {0, 1, 0, -1};
        const char* dir_names[] = {"N", "E", "S", "W"};
        
        std::string solution_path;
        
        while (!q.empty()) {
            State cur = q.front(); q.pop();
            
            if (maze[cur.r][cur.c] == 'E') {
                solution_path = cur.path;
                break;
            }
            
            for (int d = 0; d < 4; ++d) {
                int nr = cur.r + dr[d], nc = cur.c + dc[d];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && 
                    maze[nr][nc] != '#' && !visited.count({nr, nc})) {
                    visited.insert({nr, nc});
                    q.push({nr, nc, d, cur.path + dir_names[d]});
                }
            }
        }
        
        // Train PSGN on the solution sequence
        std::ostringstream oss;
        oss << "MAZE_START ";
        for (char c : solution_path) oss << c << " ";
        oss << "MAZE_END .";
        
        std::string training_data = oss.str();
        auto tokens = tokenize(training_data);
        
        for (const auto& tok : tokens) {
            int id = encoder.get_node_id(tok);
            if (tok == ".") {
                wm.clear();
                continue;
            }
            
            const auto& sdr = encoder.encode(tok);
            wm.hold(sdr);
            
            // Build graph edges
            if (wm.read()) {
                // Create transitions between consecutive tokens
            }
        }
        
        // Direct graph training
        std::vector<std::string> seq = {"MAZE_START"};
        for (char c : solution_path) seq.emplace_back(1, c);
        seq.push_back("MAZE_END");
        
        for (size_t i = 0; i + 1 < seq.size(); ++i) {
            int from = encoder.get_node_id(seq[i]);
            int to = encoder.get_node_id(seq[i+1]);
            auto sdr = encoder.encode(seq[i]);
            graph.set_edge_weight(from, to, 1.0f, nullptr, &sdr);
        }
    }
    
    std::string solve() {
        // Use learned graph to navigate
        std::string path;
        int cur_id = encoder.get_node_id("MAZE_START");
        
        for (int step = 0; step < 100; ++step) {
            auto preds = graph.get_predictions(cur_id, nullptr);
            if (preds.empty()) break;
            
            std::string next_tok = encoder.get_token(preds[0].node_id);
            if (next_tok == "MAZE_END") break;
            if (next_tok.length() == 1 && next_tok[0] >= 'A' && next_tok[0] <= 'Z') {
                path += next_tok[0];
            }
            cur_id = preds[0].node_id;
        }
        
        return path;
    }
    
    void print_maze() {
        std::cout << "\n=== Maze ===\n";
        for (const auto& row : maze) {
            std::cout << row << "\n";
        }
        std::cout << "Start: (" << start.first << "," << start.second << ") ";
        std::cout << "End: (" << end.first << "," << end.second << ")\n";
    }
    
    void print_solution(const std::string& path) {
        std::cout << "\n=== Solution Path ===\n";
        std::cout << "Directions: " << path << "\n";
        std::cout << "Length: " << path.length() << " steps\n";
    }
};

/* ── Main PSGN Class ─────────────────────────────────────────────── */
class PSGN {
public:
    SDREncoder l1_encoder;
    PredictiveGraph l1_graph;
    SDREncoder l2_encoder;
    PredictiveGraph l2_graph;
    WorkingMemory scratchpad;
    std::vector<std::string> l1_seq_buf;
    uint8_t* current_l2_sdr;
    
    PSGN(int l1_size = 1024, double l1_sp = 0.02, int l2_size = 256, double l2_sp = 0.05)
        : l1_encoder(l1_size, l1_sp), l1_graph(l2_size, l1_size),
          l2_encoder(l2_size, l2_sp), l2_graph(l2_size, l2_size),
          scratchpad(l1_size), current_l2_sdr(nullptr) {
        l1_seq_buf.reserve(MAX_SEQ_BUF);
    }
    
    ~PSGN() {
        delete[] current_l2_sdr;
    }
    
    void read_text(const std::string& text) {
        auto tokens = tokenize(text);
        for (const auto& tok : tokens) {
            l1_encoder.get_node_id(tok);
            if (tok == ".") {
                scratchpad.clear();
                l1_seq_buf.clear();
                continue;
            }
            l1_seq_buf.push_back(tok);
            const auto& sdr = l1_encoder.encode(tok);
            scratchpad.hold(sdr);
        }
    }
    
    std::string generate(const std::string& prompt, int length) {
        auto tokens = tokenize(prompt);
        std::vector<std::string> generated = tokens;
        
        scratchpad.clear();
        l1_seq_buf.clear();
        
        // Prime with prompt
        for (const auto& tok : tokens) {
            l1_encoder.get_node_id(tok);
            const auto& sdr = l1_encoder.encode(tok);
            scratchpad.hold(sdr);
            l1_seq_buf.push_back(tok);
        }
        
        int cur_id = l1_encoder.get_node_id(tokens.back());
        
        for (int step = 0; step < length; ++step) {
            auto preds = l1_graph.get_predictions(cur_id, scratchpad.read());
            if (preds.empty()) break;
            
            std::string next_word = l1_encoder.get_token(preds[0].node_id);
            if (next_word == ".") break;
            
            generated.push_back(next_word);
            l1_encoder.get_node_id(next_word);
            const auto& sdr = l1_encoder.encode(next_word);
            scratchpad.hold(sdr);
            l1_seq_buf.push_back(next_word);
            cur_id = preds[0].node_id;
        }
        
        std::ostringstream oss;
        for (size_t i = tokens.size(); i < generated.size(); ++i) {
            if (i > tokens.size()) oss << " ";
            oss << generated[i];
        }
        return oss.str();
    }
};

} // namespace psgn

#endif // PSGN_HPP
