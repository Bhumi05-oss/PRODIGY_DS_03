#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <limits>
using namespace std;

struct Node {
    int feature;
    double threshold;
    Node* left;
    Node* right;
    int label;
    bool isLeaf;

    Node() : feature(-1), threshold(0.0), left(nullptr), right(nullptr), label(-1), isLeaf(false) {}
};

vector<vector<double>> X;
vector<int> y;

void readCSV(string filename) {
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<double> row;
        while (getline(ss, value, ',')) {
            row.push_back(stod(value));
        }
        y.push_back((int)row.back());
        row.pop_back();
        X.push_back(row);
    }
}

// Gini Impurity
double gini(const vector<int>& labels) {
    int count0 = 0, count1 = 0;
    for (int label : labels) {
        if (label == 0) count0++;
        else count1++;
    }
    double p0 = (double)count0 / labels.size();
    double p1 = (double)count1 / labels.size();
    return 1.0 - (p0 * p0 + p1 * p1);
}

// Split dataset
void splitDataset(const vector<vector<double>>& X, const vector<int>& y, int feature, double threshold,
                  vector<vector<double>>& X_left, vector<int>& y_left,
                  vector<vector<double>>& X_right, vector<int>& y_right) {
    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i][feature] <= threshold) {
            X_left.push_back(X[i]);
            y_left.push_back(y[i]);
        } else {
            X_right.push_back(X[i]);
            y_right.push_back(y[i]);
        }
    }
}

// Find best split
void bestSplit(const vector<vector<double>>& X, const vector<int>& y, int& bestFeature, double& bestThreshold) {
    double bestGini = numeric_limits<double>::max();
    int n_features = X[0].size();

    for (int feature = 0; feature < n_features; ++feature) {
        for (size_t i = 0; i < X.size(); ++i) {
            double threshold = X[i][feature];
            vector<vector<double>> X_left, X_right;
            vector<int> y_left, y_right;

            splitDataset(X, y, feature, threshold, X_left, y_left, X_right, y_right);

            if (y_left.empty() || y_right.empty()) continue;

            double gini_left = gini(y_left);
            double gini_right = gini(y_right);
            double weighted_gini = (y_left.size() * gini_left + y_right.size() * gini_right) / y.size();

            if (weighted_gini < bestGini) {
                bestGini = weighted_gini;
                bestFeature = feature;
                bestThreshold = threshold;
            }
        }
    }
}

// Majority class
int majorityClass(const vector<int>& y) {
    int count0 = 0, count1 = 0;
    for (int val : y) {
        if (val == 0) count0++;
        else count1++;
    }
    return (count1 > count0) ? 1 : 0;
}

// Build tree
Node* buildTree(const vector<vector<double>>& X, const vector<int>& y, int depth = 0, int max_depth = 5) {
    Node* node = new Node();

    if (y.empty()) return nullptr;

    // Stopping conditions
    if (gini(y) == 0.0 || depth >= max_depth) {
        node->isLeaf = true;
        node->label = y[0];
        return node;
    }

    int feature;
    double threshold;
    bestSplit(X, y, feature, threshold);

    vector<vector<double>> X_left, X_right;
    vector<int> y_left, y_right;

    splitDataset(X, y, feature, threshold, X_left, y_left, X_right, y_right);

    if (X_left.empty() || X_right.empty()) {
        node->isLeaf = true;
        node->label = majorityClass(y);
        return node;
    }

    node->feature = feature;
    node->threshold = threshold;
    node->left = buildTree(X_left, y_left, depth + 1, max_depth);
    node->right = buildTree(X_right, y_right, depth + 1, max_depth);

    return node;
}

// Predict
int predict(Node* node, const vector<double>& sample) {
    if (node->isLeaf) return node->label;

    if (sample[node->feature] <= node->threshold)
        return predict(node->left, sample);
    else
        return predict(node->right, sample);
}

int main() {
    readCSV("bank_data.csv");

    Node* tree = buildTree(X, y);

    // Example prediction
    vector<double> sample = {35, 1, 0, 2, 300}; // Example input
    int prediction = predict(tree, sample);
    cout << "Prediction: " << prediction << endl;

    return 0;
}