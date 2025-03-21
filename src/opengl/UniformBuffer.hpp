#pragma once
#include "utils/Resource.hpp"
#include <cstddef>

class UniformBuffer : public Resource {
private:
    unsigned m_renderID = 0;
public:
    UniformBuffer() = default;
    UniformBuffer(size_t size);
    ~UniformBuffer();

    void bind() const;
    void unbind() const;

    void bindingPoint(unsigned index) const;
};