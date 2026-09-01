#pragma once

#include "BinaryFile.hpp"
#include "MemoryAllocator.hpp"

namespace pankey{

    namespace FileSystem{

        enum class GGUFValueType : unsigned int{
            UInt8 = 0,
            Int8 = 1,
            UInt16 = 2,
            Int16 = 3,
            UInt32 = 4,
            Int32 = 5,
            Float32 = 6,
            Bool = 7,
            String = 8,
            Array = 9,
            UInt64 = 10,
            Int64 = 11,
            Float64 = 12
        };

        struct GGUFSizeModel{
            using ALLOCATOR_TYPE = pankey::Memory::Allocator::MemoryAllocator;
            using VALUE_TYPE = int;
            using SIZE_TYPE = int;
        };

        using GGUFSizes = pankey::DataStructure::Array::ArrayPointer<GGUFSizeModel>;

        struct GGUFTensor{
            pankey::Type::Array::CharArray m_name;
            GGUFSizes m_dimensions;
            unsigned int m_type = 0;
            unsigned long long m_offset = 0;
        };

        struct GGUFTensorModel{
            using ALLOCATOR_TYPE = pankey::Memory::Allocator::MemoryAllocator;
            using VALUE_TYPE = GGUFTensor;
            using SIZE_TYPE = int;
        };

        using GGUFTensors = pankey::DataStructure::Array::ArrayPointer<GGUFTensorModel>;

        struct GGUFModel{
            unsigned int m_version = 0;
            unsigned long long m_metadata_count = 0;
            unsigned long long m_tensor_data_offset = 0;
            pankey::Type::Array::CharArray m_architecture;
            pankey::Type::Array::CharArray m_tokenizer_model;
            GGUFTensors m_tensors;

            void clear(){
                m_version = 0;
                m_metadata_count = 0;
                m_tensor_data_offset = 0;
                m_architecture.clear();
                m_tokenizer_model.clear();
                m_tensors.clear();
            }
        };

        struct GGUFFile{
            static bool load(BinaryFile& a_file, GGUFModel& a_model){
                GGUFModel i_model;
                unsigned int i_magic = 0;
                unsigned long long i_tensor_count = 0;
                unsigned long long i_metadata_count = 0;

                if(!a_file.read(i_magic)
                    || !a_file.read(i_model.m_version)
                    || !a_file.read(i_tensor_count)
                    || !a_file.read(i_metadata_count)
                    || i_magic != 0x46554747u
                    || (i_model.m_version != 2 && i_model.m_version != 3)
                    || i_tensor_count > 1000000ull
                    || i_metadata_count > 10000000ull){
                    return false;
                }

                unsigned int i_alignment = 32;
                for(unsigned long long i = 0; i < i_metadata_count; ++i){
                    pankey::Type::Array::CharArray i_key;
                    unsigned int i_type = 0;
                    if(!readString(a_file, i_key)
                        || !a_file.read(i_type)
                        || !readMetadataValue(a_file, i_model, i_key, i_type, i_alignment)){
                        return false;
                    }
                }

                for(unsigned long long i = 0; i < i_tensor_count; ++i){
                    GGUFTensor i_tensor;
                    unsigned int i_dimension_count = 0;
                    if(!readString(a_file, i_tensor.m_name)
                        || !a_file.read(i_dimension_count)
                        || i_dimension_count > 4){
                        return false;
                    }

                    if(i_dimension_count > 0){
                        i_tensor.m_dimensions.createArray(static_cast<int>(i_dimension_count));
                        i_tensor.m_dimensions.setLength(static_cast<int>(i_dimension_count));
                    }
                    for(unsigned int j = 0; j < i_dimension_count; ++j){
                        unsigned long long i_dimension = 0;
                        if(!a_file.read(i_dimension) || i_dimension == 0 || i_dimension > 2147483647ull){
                            return false;
                        }
                        i_tensor.m_dimensions.setFast(j, static_cast<int>(i_dimension));
                    }

                    if(!a_file.read(i_tensor.m_type)
                        || !a_file.read(i_tensor.m_offset)
                        || !i_model.m_tensors.add(i_tensor)){
                        return false;
                    }
                }

                const unsigned long long i_metadata_end = a_file.position();
                if(i_alignment == 0){
                    return false;
                }
                i_model.m_metadata_count = i_metadata_count;
                i_model.m_tensor_data_offset = ((i_metadata_end + i_alignment - 1) / i_alignment) * i_alignment;
                a_model = i_model;
                return true;
            }

        private:
            static bool readString(
                BinaryFile& a_file,
                pankey::Type::Array::CharArray& a_string
            ){
                unsigned long long i_length = 0;
                if(!a_file.read(i_length) || i_length > 1048576ull){
                    return false;
                }

                a_string.createArray(static_cast<int>(i_length) + 1);
                if(!a_file.readArray(a_string, static_cast<int>(i_length))){
                    return false;
                }
                a_string.setFast(static_cast<int>(i_length), '\0');
                a_string.setLength(static_cast<int>(i_length) + 1);
                return true;
            }

            static bool isKey(
                const pankey::Type::Array::CharArray& a_key,
                const char* a_value
            ){
                int i = 0;
                while(a_value[i] != '\0'){
                    if(i >= a_key.length() - 1 || a_key.getFast(i) != a_value[i]){
                        return false;
                    }
                    ++i;
                }
                return i == a_key.length() - 1;
            }

            static bool readMetadataValue(
                BinaryFile& a_file,
                GGUFModel& a_model,
                const pankey::Type::Array::CharArray& a_key,
                unsigned int a_type,
                unsigned int& a_alignment
            ){
                if(a_type == static_cast<unsigned int>(GGUFValueType::UInt32)
                    && isKey(a_key, "general.alignment")){
                    return a_file.read(a_alignment) && a_alignment > 0;
                }
                if(a_type == static_cast<unsigned int>(GGUFValueType::String)
                    && isKey(a_key, "general.architecture")){
                    return readString(a_file, a_model.m_architecture);
                }
                if(a_type == static_cast<unsigned int>(GGUFValueType::String)
                    && isKey(a_key, "tokenizer.ggml.model")){
                    return readString(a_file, a_model.m_tokenizer_model);
                }
                return skipValue(a_file, a_type);
            }

            static bool skipValue(BinaryFile& a_file, unsigned int a_type){
                const unsigned int i_string = static_cast<unsigned int>(GGUFValueType::String);
                const unsigned int i_array = static_cast<unsigned int>(GGUFValueType::Array);
                if(a_type == i_string){
                    return skipString(a_file);
                }
                if(a_type == i_array){
                    unsigned int i_element_type = 0;
                    unsigned long long i_count = 0;
                    if(!a_file.read(i_element_type) || !a_file.read(i_count) || i_count > 10000000ull){
                        return false;
                    }
                    if(i_element_type == i_string){
                        for(unsigned long long i = 0; i < i_count; ++i){
                            if(!skipString(a_file)){
                                return false;
                            }
                        }
                        return true;
                    }
                    const int i_value_size = valueSize(i_element_type);
                    return i_value_size > 0 && a_file.skip(i_count * static_cast<unsigned long long>(i_value_size));
                }

                const int i_value_size = valueSize(a_type);
                return i_value_size > 0 && a_file.skip(static_cast<unsigned long long>(i_value_size));
            }

            static bool skipString(BinaryFile& a_file){
                unsigned long long i_length = 0;
                return a_file.read(i_length) && a_file.skip(i_length);
            }

            static int valueSize(unsigned int a_type){
                if(a_type == static_cast<unsigned int>(GGUFValueType::UInt8)
                    || a_type == static_cast<unsigned int>(GGUFValueType::Int8)
                    || a_type == static_cast<unsigned int>(GGUFValueType::Bool)){
                    return 1;
                }
                if(a_type == static_cast<unsigned int>(GGUFValueType::UInt16)
                    || a_type == static_cast<unsigned int>(GGUFValueType::Int16)){
                    return 2;
                }
                if(a_type == static_cast<unsigned int>(GGUFValueType::UInt32)
                    || a_type == static_cast<unsigned int>(GGUFValueType::Int32)
                    || a_type == static_cast<unsigned int>(GGUFValueType::Float32)){
                    return 4;
                }
                if(a_type == static_cast<unsigned int>(GGUFValueType::UInt64)
                    || a_type == static_cast<unsigned int>(GGUFValueType::Int64)
                    || a_type == static_cast<unsigned int>(GGUFValueType::Float64)){
                    return 8;
                }
                return 0;
            }
        };
    }
}