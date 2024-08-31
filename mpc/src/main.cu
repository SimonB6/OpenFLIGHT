#include <iostream>
#include <string>

#include "../include/cxxopts.hpp"
#include <json.hpp>

#include "globals.h"
#include "util/connect.h"
#include "util/Profiler.h"
#include "util/util.cuh"
// #include "gpu/matrix.cuh"
#include "mpc/Protocols.h"

int partyNum;
std::vector<AESObject *> aes_objects;
Precompute PrecomputeObject;

extern std::string *addrs;
extern BmrNet **communicationSenders;
extern BmrNet **communicationReceivers;

extern Profiler matmul_profiler;
Profiler func_profiler;
Profiler memory_profiler;
Profiler comm_profiler;
Profiler debug_profiler;
Profiler test_profiler;

nlohmann::json piranha_config;

size_t db_bytes = 0;
size_t db_layer_max_bytes = 0;
size_t db_max_bytes = 0;

int log_learning_rate = 5;

void printUsage(const char *bin);
void deleteObjects();

void exp_helper(std::function<void(int,int)> test_func, int n, int d, int exp_times, std::string desc);
template<typename T> void cosine(int n, int d);
template<typename T, typename U> void nt(int n, int d);
template<typename T, typename U> void softmax(int n, int d);
template<typename T> void aggerate(int n, int d);

int main(int argc, char** argv) {

    // Parse options -- retrieve party id and config JSON
    cxxopts::Options options("piranha", "GPU-accelerated platform for MPC computation");
    options.add_options()
        ("p,party", "Party number", cxxopts::value<int>())
        ("c,config", "Configuration file", cxxopts::value<std::string>())
        ;
    options.allow_unrecognised_options();

    auto parsed_options = options.parse(argc, argv);

    // Print help
    if (parsed_options.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    partyNum = parsed_options["party"].as<int>();

    std::ifstream input_config(parsed_options["config"].as<std::string>());
    input_config >> piranha_config;
    std::cout << "Parse input config to json." << std::endl;

    // Start memory profiler and initialize communication between parties
    memory_profiler.start();

    //XXX initializeCommunication(options.ip_file, partyNum);
    std::vector<std::string> party_ips;
    for (int i = 0; i < 2; i++) {
	    party_ips.push_back(piranha_config["party_ips"][i]);
        std::cout << piranha_config["party_ips"][i] << std::endl;
    }
    initializeCommunication(party_ips, partyNum, 2);
    std::cout << "Communication established." << std::endl;

    synchronize(10000, 2); // wait for everyone to show up :)
    
    aes_objects.resize(2);
    for (size_t i = 0; i < 2; i++) {
        // --------------> AES_TODO
        //Get AES strings from file and create vector of AESObjects
        //options.aes_file;
        std::stringstream ss;
        ss << "files/aeskey" << i;
        std::string str = ss.str();
        str.erase(str.begin()), str.erase(str.end()-1);
        std::cout << const_cast<char *>(str.c_str()) << std::endl;
        aes_objects[i] = new AESObject(const_cast<char *>(str.c_str()));
    }

    size_t exp_times = piranha_config["trial_times"];
    size_t n = piranha_config["num_clients"];
    size_t d = piranha_config["num_dimensions"];
    size_t prime = 65537;
    using T = uint64_t;
    using U = uint8_t;
    std::cout << "----------------------------------EXP 1: cosine----------------------------------" << std::endl;
    {
        exp_helper(cosine<T>, n, d, exp_times, "cosine");
    }

    std::cout << "----------------------------------EXP 2: nt----------------------------------" << std::endl;
    {
        exp_helper(nt<T, U>, n, 1, exp_times, "nt");
    }

    std::cout << "----------------------------------EXP 3: softmax----------------------------------" << std::endl;
    {
        exp_helper(softmax<T, U>, n, 1, exp_times, "softmax");
    }

    std::cout << "----------------------------------EXP 4: aggerate----------------------------------" << std::endl;
    {
        exp_helper(aggerate<T>, n, d, exp_times, "aggerate");
    }
    

    // ----------> AES_TODO Delete AES objects
    for (int i = 0; i < aes_objects.size(); ++i) {
       delete aes_objects[i]; // Calls ~AESObject and deallocates *aes_objects[i]
    }
    aes_objects.clear();

    deleteObjects();

    // wait a bit for the prints to flush
    std::cout << std::flush;
    for(int i = 0; i < 10000000; i++);
   
    return 0;
}

void deleteObjects() {
	//close connection
	for (int i = 0; i < 2; i++) {
		if (i != partyNum) {
			delete communicationReceivers[i];
			delete communicationSenders[i];
		}
	}
	delete[] communicationReceivers;
	delete[] communicationSenders;
	delete[] addrs;
}

void exp_helper(std::function<void(int, int)> test_func, int n, int d, int exp_times, std::string desc) 
{
    comm_profiler.start();
    for (int i = 0; i < exp_times + 1; i++) {
        test_profiler.start();
        test_func(n, d);
        test_profiler.accumulate(desc);

        if (i == 0) { // sacrifice run to spin up GPU
            comm_profiler.clear();
            test_profiler.clear();
            continue;
        }
        // std::cout << "another gun" << std::endl;
    }
    std::cout << "desc: " << desc << std::endl;
    std::cout << "n: " << n << std::endl;
    std::cout << "d: " << d << std::endl;
    std::cout << "elapsed: " << test_profiler.get_elapsed(desc) / 1000.0 / exp_times << " s" << std::endl;
    std::cout << "comm transmited: " << comm_profiler.get_comm_tx_bytes() / 1024.0 / 1024.0 / exp_times << " MB" << std::endl;
    std::cout << "comm received: " << comm_profiler.get_comm_rx_bytes() / 1024.0 / 1024.0  / exp_times << " MB" << std::endl;
    std::cout << "   +-----+   " << std::endl;
    comm_profiler.clear();
    test_profiler.clear();
}

template<typename T> 
void cosine(int n, int d)
{
    // n * d \time d * 1
    TPC<T> x(n * d), y(d * 1), z(n * 1);
    x.fill(1), y.fill(1);
    matmul(
        x, y, z, 
        n, 1, d, false, false, false, (T)10);
}
template<typename T, typename U>
void nt(int n, int d)
{
    TPC<T> l(n);
    l.fill(1);
    TPC<T> m(1);
    // std::cout << "nt begin" << std::endl;
    gpu::reduceSum(l.getShare(0), m.getShare(0), true, 1, n);
    // std::cout << "nt reduce sum" << std::endl;
    TPC<T> inv_m(1), inv_l(n);
    inverse(m, inv_m);
    inverse(l, inv_l);
    // std::cout << "nt inverse" << std::endl;

    TPC<T> cos(n);
    TPC<U> sgn(n);
    cos.fill(1);
    dReLU(cos, sgn);
    // std::cout << "nt drelu" << std::endl;
    
    TPC<T> ldm(n), mdl(n);
    matmul(
        inv_m, l, ldm, 
        1, n, 1, false, false, false, (T)10);
    matmul(
        m, inv_l, mdl, 
        1, n, 1, false, false, false, (T)10);
    // std::cout << "nt matmul" << std::endl;
    selectShare(mdl, ldm, sgn, ldm);
    // std::cout << "nt selectshare" << std::endl;
}
template<typename T, typename U>
void softmax(int n, int d)
{
    TPC<T> ps(n);
    ps.fill(1);
    ps -= 1;

    // std::cout << "softmax begin" << std::endl;
    TPC<T> ps_max(1);
    TPC<U> b(n);
    int nm;
    if (n == 10) nm = 16;
    else if (n == 25) nm = 32;
    else if (n == 50) nm = 64;
    else if (n == 75) nm = 128;
    else if (n == 100) nm = 128;
    else if (n == 150) nm = 192;
    else if (n == 200) nm = 256;
    else error("invalid n");

    ps.resize(nm), b.resize(nm);
    maxpool(ps, ps_max, b, nm);
    ps.resize(n), b.resize(n);
    // std::cout << "softmax maxpool" << std::endl;

    TPC<T> z(n), zsq(n);
    z.fill(1);
    z += ps;
    gpu::elementVectorSubtract(z.getShare(0), ps_max.getShare(0), true, n, 1);
    // std::cout << "softmax elementVS" << std::endl;
    square(z, z);
    // std::cout << "softmax sqaure" << std::endl;

    TPC<T> zsum(1), inv_zsum(1);
    gpu::reduceSum(z.getShare(0), zsum.getShare(0), true, 1, n);
    // std::cout << "softmax reduceSum" << std::endl;
    inverse(zsum, inv_zsum);
    // std::cout << "softmax inverse" << std::endl;

    TPC<T> inv_zsum_vector(n);
    gpu::vectorExpand(inv_zsum.getShare(0), inv_zsum_vector.getShare(0), n);
    // std::cout << "softmax ve" << std::endl;
    z *= inv_zsum_vector;
    // std::cout << "softmax mult" << std::endl;

    TPC<T> zero_vector(n), result(n);
    zero_vector.fill(0);
    selectShare(z, zero_vector, b, result);
    // std::cout << "softmax SS" << std::endl;
}
template<typename T>
void aggerate(int n, int d)
{
    TPC<T> x(n), y(n), w(d), res(n * d);
    x.fill(1), y.fill(1), w.fill(1);
    // n \times n
    x *= y;
    // n * 1 \times 1 * d
    matmul(
        x, w, res, 
        n, d, 1, false, false, false, (T)10);
}
