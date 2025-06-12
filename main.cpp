// decision_tree_sfml.cpp

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <SFML/Graphics.hpp>   // link with -lsfml-graphics -lsfml-window -lsfml-system
#include <cfloat> // For FLT_MAX
#include <iomanip>
#include <sstream>

using namespace std;
void printIndent(int depth) {
    for (int i = 0; i < depth; ++i) cout << "  ";
}
// ————————————————————————————————————————————————————————————————————————————————
// DataSheet: reads a CSV (comma-delimited) into a 2D vector<string> and computes overall entropy.
// ————————————————————————————————————————————————————————————————————————————————
class DataSheet {
public:
    explicit DataSheet(fstream &file)
        : dataFile{}, entropyOfDatas(0.0)
    {
        readFile(file);
        entropyOfDatas = calculateEntropy();
    }
    double calculateEntropy() {
        unordered_map<string,int> classCount;
        int row = static_cast<int>(dataFile.size());
        int col = static_cast<int>(dataFile[0].size());

        for (int i = 1; i < row; ++i) {
            const string &lab = dataFile[i][col - 1];
            classCount[lab]++;
        }
        double ent = 0.0;
        for (auto const &kv : classCount) {
            double p = static_cast<double>(kv.second) / (row - 1);
            ent += -p * log2(p);
        }
        return ent;
    }
    double calculateEntropy(const vector<string>& labels, int depth) {
        unordered_map<string,int> freq;
        for (auto& lab : labels) freq[lab]++;
        double entropy = 0.0;
        int n = labels.size();

        printIndent(depth);
        cout << "Entropy calc for ";
        for (auto& kv : freq) cout << kv.first << ":" << kv.second << " ";
        cout << "→ ";

        for (auto& kv : freq) {
            double p = double(kv.second) / n;
            entropy -= p * log2(p);
        }

        cout << fixed << setprecision(3) << entropy << "\n";
        return entropy;
    }
    void printData() const {
        cout << "There are " << dataFile[0].size() << " attributes and "
             << (dataFile.size() - 1) << " data rows.\n\n";

        for (size_t j = 0; j < dataFile[0].size(); ++j) {
            cout << dataFile[0][j] << (j + 1 == dataFile[0].size() ? "\n" : ", ");
        }
        for (size_t i = 1; i < dataFile.size(); ++i) {
            for (size_t j = 0; j < dataFile[i].size(); ++j) {
                cout << dataFile[i][j] << (j + 1 == dataFile[i].size() ? "\n" : ", ");
            }
        }
        cout << "\n";
    }

    double calculateInformationGain(string_view attributeName) const {
        int attributeIndex = -1;
        int columnCount = static_cast<int>(dataFile[0].size());
        int rowCount = static_cast<int>(dataFile.size());

        for (int i = 0; i < columnCount - 1; ++i) {
            if (dataFile[0][i] == attributeName) {
                attributeIndex = i;
                break;
            }
        }
        if (attributeIndex < 0) {
            cerr << "Attribute not found: " << attributeName << "\n";
            return 0.0;
        }

        unordered_map<string, vector<vector<string>>> partitions;
        for (int i = 1; i < rowCount; ++i) {
            const string &key = dataFile[i][attributeIndex];
            partitions[key].push_back(dataFile[i]);
        }

        double infoGain = entropyOfDatas;

        for (auto const &kv : partitions) {
            const auto &subset = kv.second;
            unordered_map<string,int> labelCount;
            for (auto const &row : subset) {
                const string &lab = row[columnCount - 1];
                labelCount[lab]++;
            }
            double subsetEntropy = 0.0;
            for (auto const &lp : labelCount) {
                double p = static_cast<double>(lp.second) / subset.size();
                subsetEntropy += -p * log2(p);
            }
            infoGain -= (static_cast<double>(subset.size()) / (rowCount - 1)) * subsetEntropy;
        }

        return infoGain;
    }

    const vector<vector<string>>& getData() const {
        return dataFile;
    }

    vector<string> getHeaders() const {
        return dataFile.empty() ? vector<string>{} : dataFile[0];
    }

    double getEntropy() const {
        return entropyOfDatas;
    }

private:
    vector<vector<string>> dataFile;
    double entropyOfDatas;

    void splitDelimiter(const string &input, vector<string> &output, char delimiter) {
        output.clear();
        size_t start = 0;
        while (true) {
            size_t pos = input.find(delimiter, start);
            if (pos == string::npos) {
                output.emplace_back(input.substr(start));
                break;
            }
            output.emplace_back(input.substr(start, pos - start));
            start = pos + 1;
        }
    }

    void readFile(fstream &file) {
        string line;
        vector<string> temp;
        while (getline(file, line)) {
            splitDelimiter(line, temp, ',');
            dataFile.emplace_back(temp);
        }
    }
};

// ————————————————————————————————————————————————————————————————————————————————
// TreeNode: each node holds either an attribute (internal node) or a label (leaf).
// We also store an (x,y) for SFML drawing, and for each child, the edge value.
// ————————————————————————————————————————————————————————————————————————————————
struct TreeNode {
    string attribute;     // if internal node
    string label;         // non-empty only if leaf
    unordered_map<string, TreeNode*> children;
    sf::Vector2f position; // for visualization

    TreeNode(const string &attr, const string &lab)
        : attribute(attr), label(lab), position({0,0}) {}
};

// ————————————————————————————————————————————————————————————————————————————————
// DecisionTree: builds recursively on subsets, prints text, visualizes via SFML, and predicts.
// ————————————————————————————————————————————————————————————————————————————————
class DecisionTree {
public:
    explicit DecisionTree(DataSheet *data)
        : dataFile(data)
    {
        headers = dataFile->getHeaders();
        root = buildTree(dataFile->getData(), headers);
    }

    // Print tree textually
    void printTree(TreeNode *node = nullptr, const string &indent = "", const string &edgeValue = "", const string &path = "") const {
        if (!node) {
            if (!root) {
                cout << "Tree is empty.\n";
                return;
            }
            node = root;
        }

        string fullPath = path;
        if (!edgeValue.empty()) {
            if (!fullPath.empty()) fullPath += " -> ";
            fullPath += edgeValue;
        }

        if (!node->label.empty()) {
            cout << indent << "├── " << fullPath << ": Leaf = " << node->label << "\n";
            return;
        }

        if (!edgeValue.empty()) {
            cout << indent << "├── " << edgeValue << ": Attribute = " << node->attribute << "\n";
        } else {
            cout << indent << "Attribute = " << node->attribute << "\n";
        }

        vector<string> keys;
        for (const auto &kv : node->children) keys.push_back(kv.first);
        sort(keys.begin(), keys.end());

        for (const auto &val : keys) {
            TreeNode *child = node->children.at(val);
            printTree(child, indent + "│   ", val, fullPath);
        }
    }

    // Visualization with SFML
    void visualize() {
        const int windowWidth = 1200;
        const int windowHeight = 800;
        sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Decision Tree");

        sf::Font font;
        if (!font.loadFromFile("DejaVuSans.ttf")) {
            cerr << "ERROR: Could not load font \"DejaVuSans.ttf\". Place it in working directory.\n";
            return;
        }

        float xSpacing = 100.0f;
        float ySpacing = 100.0f;
        float currentX = 50.0f;
        computeNodePositions(root, 0, currentX, xSpacing, ySpacing);

        sf::FloatRect treeBounds = calculateTreeBounds(root);
        sf::View view(treeBounds);
        view.setViewport(sf::FloatRect(0, 0, 1, 1));

        bool dragging = false;
        sf::Vector2i prevMousePos;

        while (window.isOpen()) {
            sf::Event ev;
            while (window.pollEvent(ev)) {
                if (ev.type == sf::Event::Closed) {
                    window.close();
                } else if (ev.type == sf::Event::MouseWheelScrolled) {
                    float zoomFactor = (ev.mouseWheelScroll.delta > 0) ? 0.9f : 1.1f;
                    view.zoom(zoomFactor);
                } else if (ev.type == sf::Event::MouseButtonPressed && ev.mouseButton.button == sf::Mouse::Left) {
                    dragging = true;
                    prevMousePos = sf::Mouse::getPosition(window);
                } else if (ev.type == sf::Event::MouseButtonReleased && ev.mouseButton.button == sf::Mouse::Left) {
                    dragging = false;
                } else if (ev.type == sf::Event::MouseMoved && dragging) {
                    sf::Vector2i newMousePos = sf::Mouse::getPosition(window);
                    sf::Vector2f delta = window.mapPixelToCoords(prevMousePos) - window.mapPixelToCoords(newMousePos);
                    view.move(delta);
                    prevMousePos = newMousePos;
                }
            }

            window.clear(sf::Color::White);
            window.setView(view);
            drawTree(window, root, font);
            window.display();

            // After closing window, break loop
            if (!window.isOpen()) break;
        }
    }

    // ────────────────────────────────────────────────────────────────────────────────
    // Predict method: traverse tree based on input attribute values
    string predict(const unordered_map<string,string> &input) const {
        TreeNode* node = root;
        while (node && node->label.empty()) {
            auto it = input.find(node->attribute);
            if (it == input.end()) return "Unknown";
            auto childIt = node->children.find(it->second);
            if (childIt == node->children.end()) return "Unknown";
            node = childIt->second;
        }
        return node ? node->label : "Unknown";
    }

private:
    TreeNode   *root     = nullptr;
    DataSheet  *dataFile = nullptr;
    vector<string> headers;
TreeNode* buildTree(const vector<vector<string>>& data,
                    const vector<string>& headers,
                    int depth = 0) {
    int rowCount = data.size();
    int colCount = headers.size();
    int labelIdx = colCount - 1;

    // Check if all labels are the same
    const string& firstLab = data[1][labelIdx];
    bool allSame = true;
    for (int i = 2; i < rowCount; ++i) {
        if (data[i][labelIdx] != firstLab) {
            allSame = false;
            break;
        }
    }
    if (allSame) {
        printIndent(depth);
        cout << "All labels = " << firstLab << " → Leaf\n";
        return new TreeNode("", firstLab);
    }

    // If only label left, choose majority
    if (colCount <= 2) {
        unordered_map<string,int> freq;
        for (int i = 1; i < rowCount; ++i)
            freq[data[i][labelIdx]]++;
        string maj; int bestC = 0;
        for (auto& kv : freq)
            if (kv.second > bestC) maj = kv.first, bestC = kv.second;

        printIndent(depth);
        cout << "No attributes left → majority = " << maj << "\n";
        return new TreeNode("", maj);
    }

    // Select best attribute by IG
    printIndent(depth);
    cout << "Calculating gains for attributes:\n";
    int bestIdx = -1;
    double bestGain = -1.0;
    for (int i = 0; i < colCount - 1; ++i) {
        printIndent(depth);
        cout << "- Attribute \"" << headers[i] << "\":\n";
        double gain = calculateIG_OnSubset(data, i, depth+1);
        if (gain > bestGain) {
            bestGain = gain;
            bestIdx = i;
        }
    }

    // If no gain, fallback to majority
    if (bestIdx < 0) {
        unordered_map<string,int> freq;
        for (int i = 1; i < rowCount; ++i)
            freq[data[i][labelIdx]]++;
        string maj; int bestC = 0;
        for (auto& kv : freq)
            if (kv.second > bestC) maj = kv.first, bestC = kv.second;

        printIndent(depth);
        cout << "All gains ≤ 0 → majority = " << maj << "\n";
        return new TreeNode("", maj);
    }

    // Split on best attribute
    string bestAttr = headers[bestIdx];
    printIndent(depth);
    cout << "Best attribute = " << bestAttr
         << " (Gain=" << fixed << setprecision(3) << bestGain << ")\n";

    TreeNode* node = new TreeNode(bestAttr, "");

    // Partition data
    unordered_map<string, vector<vector<string>>> partitions;
    for (int i = 1; i < rowCount; ++i) {
        string val = data[i][bestIdx];
        vector<string> row = data[i];
        row.erase(row.begin() + bestIdx);
        partitions[val].push_back(row);
    }

    // New headers
    vector<string> newHeaders = headers;
    newHeaders.erase(newHeaders.begin() + bestIdx);

    // Build children
    for (auto& kv : partitions) {
        printIndent(depth);
        cout << "→ Creating subtree for " << bestAttr
             << " = " << kv.first << ":\n";
        // Build subset
        vector<vector<string>> subset;
        subset.push_back(newHeaders);
        for (auto& r : kv.second) subset.push_back(r);
        node->children[kv.first] = buildTree(subset, newHeaders, depth+1);
    }

    return node;
}
    double calculateIG_OnSubset(const vector<vector<string>>& subset,
                                int attrIdx,
                                int depth) {
        int rowCount = subset.size();
        int labelIdx = subset[0].size() - 1;

        // Gather labels
        vector<string> labels;
        for (int i = 1; i < rowCount; ++i)
            labels.push_back(subset[i][labelIdx]);

        printIndent(depth);
        cout << "Base entropy for this node:\n";
        double baseEnt = calculateEntropy(labels, depth+1);

        // Partition by attribute values
        unordered_map<string, vector<string>> parts;
        for (int i = 1; i < rowCount; ++i) {
            parts[subset[i][attrIdx]].push_back(subset[i][labelIdx]);
        }

        // Compute remainder
        double remainder = 0.0;
        for (auto& kv : parts) {
            const string& val = kv.first;
            auto& labs = kv.second;
            double weight = double(labs.size()) / labels.size();

            printIndent(depth);
            cout << "Split \"" << val << "\" (" << labs.size() << "/" << labels.size() << "):\n";
            double partEnt = calculateEntropy(labs, depth+1);
            remainder += weight * partEnt;
        }

        double gain = baseEnt - remainder;
        printIndent(depth);
        cout << "Information Gain = "
             << fixed << setprecision(3) << baseEnt
             << " - " << remainder
             << " = " << gain << "\n\n";
        return gain;
    }
    void computeNodePositions(TreeNode *node, int depth, float &currentX,
                              float xSpacing, float ySpacing)
    {
        if (!node) return;

        if (!node->label.empty()) {
            node->position.x = currentX;
            node->position.y = depth * ySpacing + 50.0f;
            currentX += xSpacing;
            return;
        }

        vector<string> keys;
        for (auto &kv : node->children)
            keys.push_back(kv.first);
        sort(keys.begin(), keys.end());

        float leftMost = FLT_MAX, rightMost = -1.0f;
        for (auto const &val : keys) {
            TreeNode *child = node->children[val];
            computeNodePositions(child, depth + 1, currentX, xSpacing, ySpacing);
            leftMost = min(leftMost, child->position.x);
            rightMost = max(rightMost, child->position.x);
        }
        node->position.x = (leftMost + rightMost) / 2.0f;
        node->position.y = depth * ySpacing + 50.0f;
    }

    sf::FloatRect calculateTreeBounds(TreeNode* node) {
        if (!node) return sf::FloatRect(0, 0, 0, 0);

        float minX = node->position.x, maxX = node->position.x;
        float minY = node->position.y, maxY = node->position.y;

        for (const auto& kv : node->children) {
            sf::FloatRect childBounds = calculateTreeBounds(kv.second);
            minX = std::min(minX, childBounds.left);
            maxX = std::max(maxX, childBounds.left + childBounds.width);
            minY = std::min(minY, childBounds.top);
            maxY = std::max(maxY, childBounds.top + childBounds.height);
        }

        return sf::FloatRect(minX - 50, minY - 50, (maxX - minX) + 100, (maxY - minY) + 100);
    }
    double calculateEntropy(const vector<string>& labels, int depth) {
    unordered_map<string,int> freq;
    for (auto& lab : labels) freq[lab]++;
    double entropy = 0.0;
    int n = labels.size();

    printIndent(depth);
    cout << "Entropy calc for ";
    for (auto& kv : freq) cout << kv.first << ":" << kv.second << " ";
    cout << "→ ";

    for (auto& kv : freq) {
        double p = double(kv.second) / n;
        entropy -= p * log2(p);
    }

    cout << fixed << setprecision(3) << entropy << "\n";
    return entropy;
}
    void drawTree(sf::RenderWindow &win, TreeNode *node, const sf::Font &font) const {
        if (!node) return;

        for (auto const &kv : node->children) {
            TreeNode *child = kv.second;
            if (!child) continue;

            sf::Vertex line[] = {
                sf::Vertex(node->position, sf::Color::Black),
                sf::Vertex(child->position, sf::Color::Black)
            };
            win.draw(line, 2, sf::Lines);

            sf::Vector2f mid = (node->position + child->position) / 2.0f;
            sf::Text edgeText;
            edgeText.setFont(font);
            edgeText.setCharacterSize(12);
            edgeText.setFillColor(sf::Color::Blue);
            edgeText.setString(kv.first);
            sf::FloatRect edgeBounds = edgeText.getLocalBounds();
            edgeText.setPosition(mid.x - edgeBounds.width / 2, mid.y - edgeBounds.height / 2);
            win.draw(edgeText);

            drawTree(win, child, font);
        }

        float radius = 20.0f;
        sf::CircleShape circle(radius);
        if (!node->label.empty())
            circle.setFillColor(sf::Color(180,255,180));
        else
            circle.setFillColor(sf::Color::White);
        circle.setOutlineColor(sf::Color::Black);
        circle.setOutlineThickness(2.0f);
        circle.setPosition(node->position.x - radius, node->position.y - radius);
        win.draw(circle);

        sf::Text text;
        text.setFont(font);
        text.setCharacterSize(14);
        text.setFillColor(sf::Color::Black);
        text.setString(node->label.empty() ? node->attribute : node->label);
        sf::FloatRect bounds = text.getLocalBounds();
        text.setPosition(
            node->position.x - bounds.width / 2.0f,
            node->position.y - bounds.height / 2.0f - 5.0f
        );
        win.draw(text);
    }
};


// Function to split a string by a delimiter
vector<string> split(const string& line, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(line);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Function to read CSV/TXT file with a given delimiter
vector<vector<string>> readTableFromFile(const string& filename, char delimiter) {
    vector<vector<string>> table;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << "\n";
        return table;
    }

    string line;
    while (getline(file, line)) {
        vector<string> row = split(line, delimiter);
        table.push_back(row);
    }

    file.close();
    return table;
}

int main() {
    string filename;
    cout << "Enter CSV or TXT file name to read: ";
    cin >> filename;

    string lowerFilename = filename;
    transform(lowerFilename.begin(), lowerFilename.end(), lowerFilename.begin(), ::tolower);

    char delimiter = ',';
    if (lowerFilename.find(".txt") != string::npos) {
        string delimStr;
        cout << "Enter the delimiter character for the TXT file (e.g. , . ; |): ";
        cin >> delimStr;
        if (!delimStr.empty()) {
            delimiter = delimStr[0];
        }
    }

    fstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << "\n";
        return 1;
    }

    DataSheet data(file);
    file.close();

    data.printData();

    DecisionTree tree(&data);
    tree.printTree();
    tree.visualize();

    char choice;
    do {
        cout << "Do you want to make a guess? (y/n): ";
        cin >> choice;
        if (choice == 'y' || choice == 'Y') {
            unordered_map<string,string> input;
            for (size_t i = 0; i < data.getHeaders().size() - 1; ++i) {
                string attr = data.getHeaders()[i];
                string val;
                cout << "Enter value for " << attr << ": ";
                cin >> val;
                input[attr] = val;
            }
            string prediction = tree.predict(input);
            cout << "Prediction: " << prediction << endl;
        }
    } while (choice == 'y' || choice == 'Y');

    return 0;
}