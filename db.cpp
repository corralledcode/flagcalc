//
// Created by peterglenn on 6/6/24.
//

#include "db.h"

#include <istream>
#include <vector>

#include "feature.h"
#include "graphs.h"
//#include <mysqlx/xdevapi.h>

class sqlfeature : public feature {

    std::string cmdlineoption() {
        return "q";
    }
    std::string cmdlineoptionlong() {
        return "SQL query";
    }

    void listoptions() override {
        feature::listoptions();
        *_os << "\t" << "i=<filename>: \t\t SQL input filename, or \"std::cin\"\n";
        *_os << "\t" << "\"o=<filename>\": \t\t SQL output filename, or \"std::cout\"\n";
    }


    sqlfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {}

    void execute(std::vector<std::string> args) override {
        std::ofstream ofs;
        std::string ifname {};
        std::string ofname {};
        std::string user {};
        std::string pwd {};
        std::string host = "localhost";
        bool ofsrequiresclose = false;
        std::vector<std::pair<std::string,std::string>> cmdlineoptions = cmdlineparseiterationtwo(args);
        for (int n = 0; n < cmdlineoptions.size(); ++n) {
            if (cmdlineoptions[n].first == "o") {
                ofname = cmdlineoptions[n].second;
                continue;
            }
            if (cmdlineoptions[n].first == "i") {
                ifname = cmdlineoptions[n].second;
                std::ifstream infile(ifname);
                if (infile.good()) {
                    std::ifstream ifs;
                    ifs.open(ifname, std::fstream::in );
                    while (!ifs.eof()) {
                        std::string tmp {};
                        ifs >> tmp;
                    }
                    ifs.close();
                    //std::cout << "Verbosity: " << verbositylevel << "\n";
                } else {
                    std::cout << "Couldn't open file for reading " << ifname << "\n";
                }
                continue;
            }
            if (cmdlineoptions[n].first == "default") {
            }
        }

        /*

        mysqlx::Session sess("localhost", 33060, "user", "password");
        mysqlx::Schema db= sess.getSchema("test");
        // or Schema db(sess, "test");

        mysqlx::Collection myColl = db.getCollection("my_collection");
        // or Collection myColl(db, "my_collection");

        mysqlx::DocResult myDocs = myColl.find("name like :param")
                                 .limit(1)
                                 .bind("param","L%").execute();

        std::cout << myDocs.fetchOne();

*/


    }
};







