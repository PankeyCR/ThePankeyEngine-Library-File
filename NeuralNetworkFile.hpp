#pragma once

#include "BinaryFile.hpp"
#include "NeuralNetwork.hpp"

namespace pankey{

    namespace FileSystem{

        inline bool saveNeuralNetwork(BinaryFile& a_file, const pankey::AI::NeuralNetwork& a_network){
            const unsigned int i_magic = 0x504E4E4Du;
            const int i_version = 3;
            const int i_layer_count = a_network.m_layer_sizes.length();
            const int i_neuron_count = a_network.m_neurons.length();
            int i_connection_count = 0;

            for(int i = 0; i < i_neuron_count; ++i){
                i_connection_count += a_network.m_neurons.getFast(i).connectionCount();
            }

            if(i_layer_count < 2 || i_neuron_count <= 0
                || (a_network.m_optimizer.m_weight_velocities.length() != 0
                    && a_network.m_optimizer.m_weight_velocities.length() != i_connection_count)
                || (a_network.m_optimizer.m_bias_velocities.length() != 0
                    && a_network.m_optimizer.m_bias_velocities.length() != i_neuron_count)
                || (a_network.m_optimizer.m_weight_squares.length() != 0
                    && a_network.m_optimizer.m_weight_squares.length() != i_connection_count)
                || (a_network.m_optimizer.m_bias_squares.length() != 0
                    && a_network.m_optimizer.m_bias_squares.length() != i_neuron_count)
                || (a_network.m_optimizer.m_type != pankey::AI::OptimizerType::Adam
                    && (!a_network.m_optimizer.m_weight_squares.isEmpty()
                        || !a_network.m_optimizer.m_bias_squares.isEmpty()))
                || (a_network.m_optimizer.m_type == pankey::AI::OptimizerType::Adam
                    && ((a_network.m_optimizer.m_weight_velocities.length() == 0)
                        != (a_network.m_optimizer.m_weight_squares.length() == 0)))
                || !a_file.write(i_magic)
                || !a_file.write(i_version)
                || !a_file.write(i_layer_count)){
                return false;
            }

            for(int i = 0; i < i_layer_count; ++i){
                if(!a_file.write(a_network.m_layer_sizes.getFast(i))){
                    return false;
                }
            }

            if(!a_file.write(i_neuron_count)){
                return false;
            }

            for(int i = 0; i < i_neuron_count; ++i){
                const pankey::AI::Neuron i_neuron = a_network.m_neurons.getFast(i);
                const int i_activation = static_cast<int>(i_neuron.m_activation);
                const int i_connection_count = i_neuron.connectionCount();

                if(!a_file.write(i_neuron.m_bias)
                    || !a_file.write(i_activation)
                    || !a_file.write(i_connection_count)){
                    return false;
                }

                for(int j = 0; j < i_connection_count; ++j){
                    if(!a_file.write(i_neuron.m_targets.getFast(j))
                        || !a_file.write(i_neuron.m_weights.getFast(j))){
                        return false;
                    }
                }
            }

            const int i_optimizer_type = static_cast<int>(a_network.m_optimizer.m_type);
            const int i_weight_velocity_count = a_network.m_optimizer.m_weight_velocities.length();
            const int i_bias_velocity_count = a_network.m_optimizer.m_bias_velocities.length();
            const int i_weight_square_count = a_network.m_optimizer.m_weight_squares.length();
            const int i_bias_square_count = a_network.m_optimizer.m_bias_squares.length();
            if(!a_file.write(i_optimizer_type)
                || !a_file.write(a_network.m_optimizer.m_momentum)
                || !a_file.write(a_network.m_optimizer.m_beta_1)
                || !a_file.write(a_network.m_optimizer.m_beta_2)
                || !a_file.write(a_network.m_optimizer.m_epsilon)
                || !a_file.write(a_network.m_optimizer.m_step)
                || !a_file.write(a_network.m_optimizer.m_beta_1_power)
                || !a_file.write(a_network.m_optimizer.m_beta_2_power)
                || !a_file.write(i_weight_velocity_count)
                || !a_file.write(i_bias_velocity_count)
                || !a_file.write(i_weight_square_count)
                || !a_file.write(i_bias_square_count)){
                return false;
            }
            for(int i = 0; i < i_weight_velocity_count; ++i){
                if(!a_file.write(a_network.m_optimizer.m_weight_velocities.getFast(i))){
                    return false;
                }
            }
            for(int i = 0; i < i_bias_velocity_count; ++i){
                if(!a_file.write(a_network.m_optimizer.m_bias_velocities.getFast(i))){
                    return false;
                }
            }
            for(int i = 0; i < i_weight_square_count; ++i){
                if(!a_file.write(a_network.m_optimizer.m_weight_squares.getFast(i))){
                    return false;
                }
            }
            for(int i = 0; i < i_bias_square_count; ++i){
                if(!a_file.write(a_network.m_optimizer.m_bias_squares.getFast(i))){
                    return false;
                }
            }

            return true;
        }

        inline bool loadNeuralNetworkData(BinaryFile& a_file, pankey::AI::NeuralNetwork& a_network){
            const unsigned int i_expected_magic = 0x504E4E4Du;
            unsigned int i_magic = 0;
            int i_version = 0;
            int i_layer_count = 0;

            if(!a_file.read(i_magic)
                || !a_file.read(i_version)
                || !a_file.read(i_layer_count)
                || i_magic != i_expected_magic
                || (i_version != 1 && i_version != 2 && i_version != 3)
                || i_layer_count < 2){
                return false;
            }

            pankey::AI::TensorSizes i_layer_sizes;
            for(int i = 0; i < i_layer_count; ++i){
                int i_layer_size = 0;
                if(!a_file.read(i_layer_size) || i_layer_size <= 0 || !i_layer_sizes.add(i_layer_size)){
                    return false;
                }
            }

            int i_neuron_count = 0;
            if(!a_file.read(i_neuron_count) || !a_network.create(i_layer_sizes) || i_neuron_count != a_network.m_neurons.length()){
                return false;
            }

            for(int i = 0; i < i_neuron_count; ++i){
                float i_bias = 0.0f;
                int i_activation = 0;
                int i_connection_count = 0;
                if(!a_file.read(i_bias)
                    || !a_file.read(i_activation)
                    || !a_file.read(i_connection_count)
                    || i_connection_count < 0
                    || i_connection_count > i_neuron_count
                    || i_activation < static_cast<int>(pankey::AI::Activation::Linear)
                    || i_activation > static_cast<int>(pankey::AI::Activation::LeakyRelu)
                    || !a_network.setBias(i, i_bias)
                    || !a_network.setActivation(i, static_cast<pankey::AI::Activation>(i_activation))){
                    return false;
                }

                for(int j = 0; j < i_connection_count; ++j){
                    int i_target = 0;
                    float i_weight = 0.0f;
                    if(!a_file.read(i_target)
                        || !a_file.read(i_weight)
                        || !a_network.connect(i, i_target, i_weight)){
                        return false;
                    }
                }
            }

            if(!a_network.compile()){
                return false;
            }

            if(i_version == 1){
                return true;
            }

            int i_optimizer_type = 0;
            float i_momentum = 0.0f;
            float i_beta_1 = 0.9f;
            float i_beta_2 = 0.999f;
            float i_epsilon = 0.00000001f;
            int i_step = 0;
            float i_beta_1_power = 1.0f;
            float i_beta_2_power = 1.0f;
            int i_weight_velocity_count = 0;
            int i_bias_velocity_count = 0;
            int i_weight_square_count = 0;
            int i_bias_square_count = 0;
            if(!a_file.read(i_optimizer_type)
                || !a_file.read(i_momentum)
                || (i_version == 3 && (!a_file.read(i_beta_1)
                    || !a_file.read(i_beta_2)
                    || !a_file.read(i_epsilon)
                    || !a_file.read(i_step)
                    || !a_file.read(i_beta_1_power)
                    || !a_file.read(i_beta_2_power)))
                || !a_file.read(i_weight_velocity_count)
                || !a_file.read(i_bias_velocity_count)
                || (i_version == 3 && (!a_file.read(i_weight_square_count)
                    || !a_file.read(i_bias_square_count)))
                || i_optimizer_type < static_cast<int>(pankey::AI::OptimizerType::SGD)
                || i_optimizer_type > static_cast<int>(i_version == 2
                    ? pankey::AI::OptimizerType::Momentum
                    : pankey::AI::OptimizerType::Adam)
                || i_momentum < 0.0f
                || i_momentum >= 1.0f
                || i_beta_1 < 0.0f
                || i_beta_1 >= 1.0f
                || i_beta_2 < 0.0f
                || i_beta_2 >= 1.0f
                || i_epsilon <= 0.0f
                || i_step < 0
                || i_beta_1_power <= 0.0f
                || i_beta_1_power > 1.0f
                || i_beta_2_power <= 0.0f
                || i_beta_2_power > 1.0f
                || i_weight_velocity_count < 0
                || i_bias_velocity_count < 0
                || i_weight_square_count < 0
                || i_bias_square_count < 0
                || ((i_weight_velocity_count == 0) != (i_bias_velocity_count == 0))
                || (i_weight_velocity_count != 0
                    && (i_weight_velocity_count != a_network.m_compiled.m_weights.length()
                        || i_bias_velocity_count != a_network.m_compiled.m_biases.length()))
                || (i_version == 3
                    && ((i_weight_square_count == 0) != (i_bias_square_count == 0)
                        || (i_weight_square_count != 0
                            && (i_weight_square_count != i_weight_velocity_count
                                || i_bias_square_count != i_bias_velocity_count))))){
                return false;
            }

            const pankey::AI::OptimizerType i_optimizer = static_cast<pankey::AI::OptimizerType>(i_optimizer_type);
            if((i_optimizer == pankey::AI::OptimizerType::Adam
                    && !a_network.m_optimizer.configureAdam(i_beta_1, i_beta_2, i_epsilon))
                || (i_optimizer != pankey::AI::OptimizerType::Adam
                    && !a_network.m_optimizer.configure(i_optimizer, i_momentum))){
                return false;
            }

            if(i_weight_velocity_count == 0){
                return true;
            }

            a_network.m_optimizer.m_weight_velocities.createArray(i_weight_velocity_count);
            a_network.m_optimizer.m_weight_velocities.setLength(i_weight_velocity_count);
            a_network.m_optimizer.m_bias_velocities.createArray(i_bias_velocity_count);
            a_network.m_optimizer.m_bias_velocities.setLength(i_bias_velocity_count);
            for(int i = 0; i < i_weight_velocity_count; ++i){
                float i_velocity = 0.0f;
                if(!a_file.read(i_velocity)){
                    return false;
                }
                a_network.m_optimizer.m_weight_velocities.setFast(i, i_velocity);
            }
            for(int i = 0; i < i_bias_velocity_count; ++i){
                float i_velocity = 0.0f;
                if(!a_file.read(i_velocity)){
                    return false;
                }
                a_network.m_optimizer.m_bias_velocities.setFast(i, i_velocity);
            }

            if(i_version == 3 && i_weight_square_count != 0){
                a_network.m_optimizer.m_weight_squares.createArray(i_weight_square_count);
                a_network.m_optimizer.m_weight_squares.setLength(i_weight_square_count);
                a_network.m_optimizer.m_bias_squares.createArray(i_bias_square_count);
                a_network.m_optimizer.m_bias_squares.setLength(i_bias_square_count);
                for(int i = 0; i < i_weight_square_count; ++i){
                    float i_square = 0.0f;
                    if(!a_file.read(i_square) || i_square < 0.0f){
                        return false;
                    }
                    a_network.m_optimizer.m_weight_squares.setFast(i, i_square);
                }
                for(int i = 0; i < i_bias_square_count; ++i){
                    float i_square = 0.0f;
                    if(!a_file.read(i_square) || i_square < 0.0f){
                        return false;
                    }
                    a_network.m_optimizer.m_bias_squares.setFast(i, i_square);
                }
            }

            if(i_version == 3){
                a_network.m_optimizer.m_step = i_step;
                a_network.m_optimizer.m_beta_1_power = i_beta_1_power;
                a_network.m_optimizer.m_beta_2_power = i_beta_2_power;
            }

            return true;
        }

        inline bool loadNeuralNetwork(BinaryFile& a_file, pankey::AI::NeuralNetwork& a_network){
            pankey::AI::NeuralNetwork i_loaded_network;
            if(!loadNeuralNetworkData(a_file, i_loaded_network)){
                return false;
            }
            a_network = i_loaded_network;
            return true;
        }
    }
}