#pragma once

#include "GGUFFile.hpp"
#include "BrainNetworkPolicy.hpp"

namespace pankey{

    namespace FileSystem{

        struct NeuralNetworkGGUFFile{

            template<class Policy>
            static bool save(
                BinaryFile& a_file,
                const typename Policy::BRAIN_TYPE& a_brain,
                const typename Policy::NETWORK_TYPE& a_network
            ){
                const unsigned int i_magic = 0x46554747u;
                const unsigned int i_version = 3;
                const unsigned long long i_tensor_count = 6;
                const unsigned long long i_metadata_count = 2;
                const unsigned int i_alignment = 32;
                const int i_layers = a_network.m_layerSizes.length();
                const int i_nodes = a_brain.m_biases.length();
                const int i_connections = a_network.m_source.length();

                if(i_layers <= 0 || i_nodes <= 0
                    || a_network.m_target.length() != i_connections
                    || a_network.m_weightIndex.length() != i_connections
                    || a_brain.m_activations.length() != i_nodes
                    || !a_file.write(i_magic)
                    || !a_file.write(i_version)
                    || !a_file.write(i_tensor_count)
                    || !a_file.write(i_metadata_count)
                    || !writeMetadataString(a_file, "general.architecture", "pankey.sparse_network")
                    || !writeMetadataUInt32(a_file, "general.alignment", i_alignment)){
                    return false;
                }

                unsigned long long i_offset = 0;
                if(!writeTensorInfo(a_file, "pankey.layer_sizes", 26, i_layers, i_offset, i_alignment)
                    || !writeTensorInfo(a_file, "pankey.biases", 0, i_nodes, i_offset, i_alignment)
                    || !writeTensorInfo(a_file, "pankey.activations", 0, i_nodes, i_offset, i_alignment)
                    || !writeTensorInfo(a_file, "pankey.sources", 26, i_connections, i_offset, i_alignment)
                    || !writeTensorInfo(a_file, "pankey.targets", 26, i_connections, i_offset, i_alignment)
                    || !writeTensorInfo(a_file, "pankey.weights", 0, a_brain.m_weights.length(), i_offset, i_alignment)){
                    return false;
                }

                const unsigned long long i_data_start = align(a_file.position(), i_alignment);
                return writePadding(a_file, i_data_start - a_file.position())
                    && writeIntegers(a_file, a_network.m_layerSizes, i_alignment)
                    && writeFloats(a_file, a_brain.m_biases, i_alignment)
                    && writeFloats(a_file, a_brain.m_activations, i_alignment)
                    && writeIntegers(a_file, a_network.m_source, i_alignment)
                    && writeIntegers(a_file, a_network.m_target, i_alignment)
                    && writeFloats(a_file, a_brain.m_weights, i_alignment);
            }

            template<class Policy>
            static bool load(
                BinaryFile& a_file,
                typename Policy::BRAIN_TYPE& a_brain,
                typename Policy::NETWORK_TYPE& a_network
            ){
                GGUFModel i_model;
                if(!GGUFFile::load(a_file, i_model)
                    || !matches(i_model.m_architecture, "pankey.sparse_network")){
                    return false;
                }

                const int i_layers = findTensor(i_model, "pankey.layer_sizes");
                const int i_biases = findTensor(i_model, "pankey.biases");
                const int i_activations = findTensor(i_model, "pankey.activations");
                const int i_sources = findTensor(i_model, "pankey.sources");
                const int i_targets = findTensor(i_model, "pankey.targets");
                const int i_weights = findTensor(i_model, "pankey.weights");
                if(i_layers < 0 || i_biases < 0 || i_activations < 0
                    || i_sources < 0 || i_targets < 0 || i_weights < 0){
                    return false;
                }

                pankey::Type::Array::IntegerArray i_layerValues;
                pankey::Type::Array::IntegerArray i_sourcesValues;
                pankey::Type::Array::IntegerArray i_targetsValues;
                pankey::Type::Array::IntegerArray i_weightIndexes;
                pankey::Type::Array::FloatArray i_biasValues;
                pankey::Type::Array::FloatArray i_activationValues;
                pankey::Type::Array::FloatArray i_weightValues;
                if(!readIntegers(a_file, i_model, i_layers, i_layerValues)
                    || !readFloats(a_file, i_model, i_biases, i_biasValues)
                    || !readFloats(a_file, i_model, i_activations, i_activationValues)
                    || !readIntegers(a_file, i_model, i_sources, i_sourcesValues)
                    || !readIntegers(a_file, i_model, i_targets, i_targetsValues)
                    || !readFloats(a_file, i_model, i_weights, i_weightValues)
                    || i_biasValues.length() != i_activationValues.length()
                    || i_sourcesValues.length() != i_targetsValues.length()
                    || i_sourcesValues.length() != i_weightValues.length()){
                    return false;
                }
                i_weightIndexes.createArray(i_weightValues.length());
                i_weightIndexes.setLength(i_weightValues.length());
                for(int i = 0; i < i_weightIndexes.length(); ++i){
                    i_weightIndexes.setFast(i, i);
                }

                typename Policy::NETWORK_TYPE i_network;
                if(!pankey::AI::compileNetwork(
                    typename Policy::COMPILATION_MODEL{},
                    i_network,
                    i_layerValues,
                    i_sourcesValues,
                    i_targetsValues,
                    i_weightIndexes
                )){
                    return false;
                }

                typename Policy::BRAIN_TYPE i_brain;
                if(!i_brain.create(
                    static_cast<uint32_t>(i_biasValues.length()),
                    static_cast<uint32_t>(i_weightValues.length()))){
                    return false;
                }
                for(int i = 0; i < i_biasValues.length(); ++i){
                    i_brain.m_biases.setFast(i, i_biasValues.getFast(i));
                    i_brain.m_activations.setFast(i, i_activationValues.getFast(i));
                }
                for(int i = 0; i < i_weightValues.length(); ++i){
                    i_brain.m_weights.setFast(i, i_weightValues.getFast(i));
                }

                a_network = i_network;
                a_brain = i_brain;
                return true;
            }

        private:
            static unsigned long long align(unsigned long long a_value, unsigned int a_alignment){
                return ((a_value + a_alignment - 1) / a_alignment) * a_alignment;
            }

            static bool writeText(BinaryFile& a_file, const char* a_text){
                unsigned long long i_length = 0;
                while(a_text[i_length] != '\0'){
                    ++i_length;
                }
                if(!a_file.write(i_length)){
                    return false;
                }
                for(unsigned long long i = 0; i < i_length; ++i){
                    if(!a_file.write(a_text[i])){
                        return false;
                    }
                }
                return true;
            }

            static bool writeMetadataString(BinaryFile& a_file, const char* a_key, const char* a_value){
                const unsigned int i_type = 8;
                return writeText(a_file, a_key) && a_file.write(i_type) && writeText(a_file, a_value);
            }

            static bool writeMetadataUInt32(BinaryFile& a_file, const char* a_key, unsigned int a_value){
                const unsigned int i_type = 4;
                return writeText(a_file, a_key) && a_file.write(i_type) && a_file.write(a_value);
            }

            static bool writeTensorInfo(BinaryFile& a_file, const char* a_name, unsigned int a_type, int a_count, unsigned long long& a_offset, unsigned int a_alignment){
                const unsigned int i_dimensions = 1;
                const unsigned long long i_count = static_cast<unsigned long long>(a_count);
                if(a_count <= 0 || !writeText(a_file, a_name)
                    || !a_file.write(i_dimensions) || !a_file.write(i_count)
                    || !a_file.write(a_type) || !a_file.write(a_offset)){
                    return false;
                }
                a_offset += align(i_count * 4ull, a_alignment);
                return true;
            }

            static bool writePadding(BinaryFile& a_file, unsigned long long a_size){
                const unsigned char i_zero = 0;
                for(unsigned long long i = 0; i < a_size; ++i){
                    if(!a_file.write(i_zero)){
                        return false;
                    }
                }
                return true;
            }

            template<class ArrayType>
            static bool writeIntegers(BinaryFile& a_file, const ArrayType& a_values, unsigned int a_alignment){
                for(int i = 0; i < a_values.length(); ++i){
                    if(!a_file.write(a_values.getFast(i))){
                        return false;
                    }
                }
                return writePadding(a_file, align(static_cast<unsigned long long>(a_values.length()) * 4ull, a_alignment) - static_cast<unsigned long long>(a_values.length()) * 4ull);
            }

            template<class ArrayType>
            static bool writeFloats(BinaryFile& a_file, const ArrayType& a_values, unsigned int a_alignment){
                for(int i = 0; i < a_values.length(); ++i){
                    if(!a_file.write(a_values.getFast(i))){
                        return false;
                    }
                }
                return writePadding(a_file, align(static_cast<unsigned long long>(a_values.length()) * 4ull, a_alignment) - static_cast<unsigned long long>(a_values.length()) * 4ull);
            }

            static bool matches(const pankey::Type::Array::CharArray& a_value, const char* a_text){
                int i = 0;
                while(a_text[i] != '\0'){
                    if(i >= a_value.length() - 1 || a_value.getFast(i) != a_text[i]){
                        return false;
                    }
                    ++i;
                }
                return i == a_value.length() - 1;
            }

            static int findTensor(const GGUFModel& a_model, const char* a_name){
                for(int i = 0; i < a_model.m_tensors.length(); ++i){
                    if(matches(a_model.m_tensors.getFast(i).m_name, a_name)){
                        return i;
                    }
                }
                return -1;
            }

            static bool seekTensor(BinaryFile& a_file, const GGUFModel& a_model, int a_index){
                if(a_index < 0 || a_index >= a_model.m_tensors.length()){
                    return false;
                }
                const GGUFTensor i_tensor = a_model.m_tensors.getFast(a_index);
                return a_file.seek(a_model.m_tensor_data_offset + i_tensor.m_offset);
            }

            template<class ArrayType>
            static bool readIntegers(BinaryFile& a_file, const GGUFModel& a_model, int a_index, ArrayType& a_values){
                const GGUFTensor i_tensor = a_model.m_tensors.getFast(a_index);
                if(i_tensor.m_type != 26 || i_tensor.m_dimensions.length() != 1
                    || !seekTensor(a_file, a_model, a_index)){
                    return false;
                }
                a_values.createArray(i_tensor.m_dimensions.getFast(0));
                a_values.setLength(i_tensor.m_dimensions.getFast(0));
                return a_file.readArray(a_values, a_values.getSize());
            }

            template<class ArrayType>
            static bool readFloats(BinaryFile& a_file, const GGUFModel& a_model, int a_index, ArrayType& a_values){
                const GGUFTensor i_tensor = a_model.m_tensors.getFast(a_index);
                if(i_tensor.m_type != 0 || i_tensor.m_dimensions.length() != 1
                    || !seekTensor(a_file, a_model, a_index)){
                    return false;
                }
                a_values.createArray(i_tensor.m_dimensions.getFast(0));
                a_values.setLength(i_tensor.m_dimensions.getFast(0));
                return a_file.readArray(a_values, a_values.getSize());
            }
        };
    }
}
