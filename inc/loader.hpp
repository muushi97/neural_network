#ifndef IG_LOADER_HPP
#define IG_LOADER_HPP

#include <fstream>
#include <string>
#include <random>
#include <algorithm>

#include "tensor.hpp"


template <class T>
class loader {
public:
    virtual tensor<T> get_train_data(std::size_t i) = 0;
    virtual tensor<T> get_test_data(std::size_t i) = 0;
    virtual tensor<T> get_valid_data(std::size_t i) = 0;
    virtual tensor<T> get_train_label(std::size_t i) = 0;
    virtual tensor<T> get_test_label(std::size_t i) = 0;
    virtual tensor<T> get_valid_label(std::size_t i) = 0;

    virtual std::size_t train_size() const = 0;
    virtual std::size_t test_size() const = 0;
    virtual std::size_t valid_size() const = 0;

    virtual ~loader() { }
};

template <class T>
class mnist_loader : public loader<T> {
    std::ifstream train_image;
    std::ifstream train_label;
    std::ifstream test_image;
    std::ifstream test_label;

    T data_max, data_min;
    std::size_t width, height;
    std::size_t dim;

    std::vector<std::size_t> train_indices;
    std::vector<std::size_t> test_indices;
    std::vector<std::size_t> train_valid_indices;
    std::vector<std::size_t> test_valid_indices;

    std::size_t convert_bytes_to_integer(char *buf, std::size_t begin, std::size_t end) {
        std::size_t c = 0;
        if (end - begin > 0)
            for (int i = begin; i < end; i++)
                c = c << 8 | (0x00ff & static_cast<std::size_t>(buf[i]));
        else
            for (int i = end; i < begin; i++)
                c = c << 8 | (0x00ff & static_cast<std::size_t>(buf[end - i + begin]));
        return c;
    }

    // 先頭の絶対位置
    std::size_t image_seek_size(std::size_t i) const { return 16 + width * height * i; }
    std::size_t label_seek_size(std::size_t i) const { return 8 + i; }

    // 画像読み込み(移動済みの前提)
    tensor<T> load_image_2dim(std::ifstream &f) {
        char buf;
        std::size_t temp;
        tensor<T> im{width, height};
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                f.read(&buf, 1);
                temp = 0x00ff & static_cast<std::size_t>(buf);
                im(x, y) = static_cast<T>(temp) * (data_max - data_min) / 255.0 + data_min;
            }
        }
        return im;
    }
    tensor<T> load_image_1dim(std::ifstream &f) {
        char buf;
        std::size_t temp;
        tensor<T> im{width * height};
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                f.read(&buf, 1);
                temp = 0x00ff & static_cast<std::size_t>(buf);
                im(x + width * y) = static_cast<T>(temp) * (data_max - data_min) / 255.0 + data_min;
            }
        }
        return im;
    }
    // ラベル読み込み(移動済みの前提)
    std::size_t load_label(std::ifstream &f) {
        char l;
        f.read(&l, 1);
        return static_cast<std::size_t>(l);
    }

    // 整数から出力層の形式へ変換
    tensor<T> label_to_layer(std::size_t l, std::size_t label_max, std::size_t label_min, T out_high, T out_low) {
        tensor<T> lay{ label_max - label_min + 1 };
        for (std::size_t i = 0; i < lay.dim(0); i++) {
            if (i + label_min == l) lay(i) = out_high;
            else                    lay(i) = out_low;
        }
        return lay;
    }

public:
    mnist_loader(std::string train_image, std::string train_label, std::string test_image, std::string test_label, T min, T max, double train_to_valid, double test_to_valid, std::size_t dim)
            : train_image(train_image, std::ios::binary), train_label(train_label, std::ios::binary)
            , test_image(test_image, std::ios::binary), test_label(test_label, std::ios::binary) 
            , data_max(max), data_min(min), dim(dim) {
        // buffer, sizes
        char buf[16];
        std::size_t train_size_, test_size_, train_valid_size_, test_valid_size_;

        // read train images
        this->train_image.read(buf, 16);
        if (convert_bytes_to_integer(buf, 0, 4) != 2051) return;
        train_size_ = convert_bytes_to_integer(buf, 4, 8);
        width = convert_bytes_to_integer(buf, 8, 12);
        height = convert_bytes_to_integer(buf, 12, 16);
        this->train_image.seekg(0, std::ios::end);
        if (this->train_image.tellg() != 16 + train_size_ * width * height) return;

        // read train labels
        this->train_label.read(buf, 8);
        if (convert_bytes_to_integer(buf, 0, 4) != 2049) return;
        if (train_size_ != convert_bytes_to_integer(buf, 4, 8)) return;
        this->train_label.seekg(0, std::ios::end);
        if (this->train_label.tellg() != 8 + train_size_) return;

        // read test images
        this->test_image.read(buf, 16);
        if (convert_bytes_to_integer(buf, 0, 4) != 2051) return;
        test_size_ = convert_bytes_to_integer(buf, 4, 8);
        if (width != convert_bytes_to_integer(buf, 8, 12)) return;
        if (height != convert_bytes_to_integer(buf, 12, 16)) return;
        this->test_image.seekg(0, std::ios::end);
        if (this->test_image.tellg() != 16 + test_size_ * width * height) return;

        // read test labels
        this->test_label.read(buf, 8);
        if (convert_bytes_to_integer(buf, 0, 4) != 2049) return;
        if (test_size_ != convert_bytes_to_integer(buf, 4, 8)) return;
        this->test_label.seekg(0, std::ios::end);
        if (this->test_label.tellg() != 8 + test_size_) return;

        // random
        std::random_device seed_gen;
        std::mt19937_64 engine(seed_gen());
        std::uniform_int_distribution<> dist(0, 1);

        train_valid_size_ = static_cast<std::size_t>(std::round(train_to_valid * train_size_));
        test_valid_size_ = static_cast<std::size_t>(std::round(test_to_valid * test_size_));
        train_size_ -= train_valid_size_;
        test_size_ -= test_valid_size_;

        // initialize and shaffle (train and valid by train)
        train_indices.resize(train_size_);
        train_valid_indices.resize(train_valid_size_);
        for (std::size_t i = 0; i < train_size_; i++) {
            dist.param(std::uniform_int_distribution<>::param_type(0, i));
            std::size_t j = dist(engine);
            if (i != j) train_indices[i] = train_indices[j];
            train_indices[j] = i;
        }
        for (std::size_t i = 0; i < train_valid_size_; i++) {
            dist.param(std::uniform_int_distribution<>::param_type(0, i+train_size_));
            std::size_t j = dist(engine);
            if (j < train_size_) {
                if (i != j) train_valid_indices[i] = train_indices[j];
                train_indices[j] = i + train_size_;
            }
            else {
                if (i != j) train_valid_indices[i] = train_valid_indices[j - train_size_];
                train_valid_indices[j - train_size_] = i + train_size_;
            }
        }

        // initialize and shaffle (test and valid by test)
        test_indices.resize(test_size_);
        test_valid_indices.resize(test_valid_size_);
        for (std::size_t i = 0; i < test_size_; i++) {
            dist.param(std::uniform_int_distribution<>::param_type(0, i));
            std::size_t j = dist(engine);
            if (i != j) test_indices[i] = test_indices[j];
            test_indices[j] = i;
        }
        for (std::size_t i = 0; i < test_valid_size_; i++) {
            dist.param(std::uniform_int_distribution<>::param_type(0, i+test_size_));
            std::size_t j = dist(engine);
            if (j < test_size_) {
                if (i != j) test_valid_indices[i] = test_indices[j];
                test_indices[j] = i + test_size_;
            }
            else {
                if (i != j) test_valid_indices[i] = test_valid_indices[j - test_size_];
                test_valid_indices[j - test_size_] = i + test_size_;
            }
        }

        // sort indices
        std::sort(train_indices.begin()         , train_indices.end());
        std::sort(train_valid_indices.begin()   , train_valid_indices.end());
        std::sort(test_indices.begin()          , test_indices.end());
        std::sort(test_valid_indices.begin()    , test_valid_indices.end());
    }

    virtual tensor<T> get_train_data(std::size_t i) {
        if (i >= train_size()) return tensor<T>{1};
        train_image.seekg(image_seek_size(train_indices[i]));
        if      (dim == 1) return load_image_1dim(train_image);
        else if (dim == 2) return load_image_2dim(train_image);
        else               return tensor<T>{1};
    }
    virtual tensor<T> get_test_data(std::size_t i) {
        if (i >= test_size()) return tensor<T>{1};
        test_image.seekg(image_seek_size(test_indices[i]));
        if      (dim == 1) return load_image_1dim(test_image);
        else if (dim == 2) return load_image_2dim(test_image);
        else               return tensor<T>{1};
    }
    virtual tensor<T> get_valid_data(std::size_t i) {
        if (i >= valid_size()) return tensor<T>{1};
        if (i < train_valid_indices.size()) {
            train_image.seekg(image_seek_size(train_valid_indices[i]));
            if      (dim == 1) return load_image_1dim(train_image);
            else if (dim == 2) return load_image_2dim(train_image);
            else               return tensor<T>{1};
        }
        else {
            test_image.seekg(image_seek_size(test_valid_indices[i]));
            if      (dim == 1) return load_image_1dim(test_image);
            else if (dim == 2) return load_image_2dim(test_image);
            else               return tensor<T>{1};
        }
    }
    virtual tensor<T> get_train_label(std::size_t i) {
        if (i >= train_size()) return tensor<T>{1};
        std::size_t l;
        train_label.seekg(label_seek_size(train_indices[i]));
        return label_to_layer(load_label(train_label), 9, 0, 1.0, 0.0);
    }
    virtual tensor<T> get_test_label(std::size_t i) {
        if (i >= test_size()) return tensor<T>{1};
        std::size_t l;
        test_label.seekg(label_seek_size(test_indices[i]));
        return label_to_layer(load_label(test_label), 9, 0, 1.0, 0.0);
    }
    virtual tensor<T> get_valid_label(std::size_t i) {
        if (i >= valid_size()) return tensor<T>{1};
        std::size_t l;
        if (i < train_valid_indices.size()) {
            train_label.seekg(label_seek_size(train_valid_indices[i]));
            return label_to_layer(load_label(train_label), 9, 0, 1.0, 0.0);
        }
        else {
            test_label.seekg(label_seek_size(test_valid_indices[i]));
            return label_to_layer(load_label(test_label), 9, 0, 1.0, 0.0);
        }
    }

    virtual std::size_t train_size() const { return train_indices.size(); }
    virtual std::size_t test_size() const { return test_indices.size(); }
    virtual std::size_t valid_size() const { return train_valid_indices.size() + test_valid_indices.size(); }
};

#endif

