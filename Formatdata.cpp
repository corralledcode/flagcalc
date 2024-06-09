//
// Created by peterglenn on 3/13/24.
//

#ifndef HELLYTOOLCPP_FORMATDATA_CPP
#define HELLYTOOLCPP_FORMATDATA_CPP

#include <string>
#include <iostream>
#include <istream>
#include <ostream>
#include <fstream>
#include <vector>
#include <regex>
#include "Batchprocesseddata.cpp"

template<typename I, typename E, typename IBPD, typename EBPD>
class Formatdata : public Batchprocessed {

protected:
    virtual void parseexternal(std::string* str, std::vector<E>*) = 0;
    virtual void parseinternal(std::string* str, std::vector<I>*) = 0;

public:
    int size_ = 0;
    IBPD* idata = nullptr;
    EBPD* edata = nullptr;

    E lookup(I di);
    I lookup(E de);
    int size() {
        if (paused())
            return size_;
        else
            return (idata->size()<=edata->size()) ? idata->size() : edata->size();
    }

    virtual void readdata(std::istream& is) {
        auto p = paused();
        pause();
        if (!idata || !edata)
            throw std::exception();
        int s=0;

        std::vector<E> eres{};
        std::string tmp = "";
        std::string tmp2 = "";
        while ((is >> tmp) && (tmp != "END")) {
            parseexternal(&tmp, &eres);
            tmp2 += tmp + " ";
            tmp = "";
            s++;
        }
        edata->readvector(eres);
        //edata->removeduplicates();
        size_ = edata->size(); // may be larger than s due to condensed parsing
        idata->setsize(size_);
        std::vector<I> ires {};
        if (tmp2.size() > 0) {
            std::regex pat{"([\\w]+)"};

            for (std::sregex_iterator p(tmp2.begin(), tmp2.end(), pat); p != std::sregex_iterator{}; ++p) {
                std::string tmp3;
                tmp3 = (*p)[1];
                parseinternal(&tmp3, &ires);
            }
            idata->readvector(ires);
            // idata->removeduplicates(); must not remove duplicates yet... wait until setvertices has been called
        }
        if (!p) {
            resume();
            edata->resume();
            idata->resume();
        }
    }

    virtual void writedata(std::ostream& os) {
    /*
        auto sz = size();
        for (int n = 0; n < sz; ++n) {
            os << "{" << idata->getdata(n) << ":" << edata->getdata(n) << "}, ";
        }*/
        os << "hello \n";

     }

    Formatdata(IBPD& iin, EBPD& ein) : Batchprocessed() {
        size_ = iin.size();
        idata = &iin;
        edata = &ein;
    }

    Formatdata(IBPD& iin) : Batchprocessed() {
        idata = &iin;
    }
    Formatdata(int s) : Batchprocessed() {
        size_ = s;
    }
    Formatdata() : Batchprocessed() {
        idata = nullptr;
        edata = nullptr;
        size_ = 0;
    }
    Formatdata(std::istream& is) : Batchprocessed() {
        if (idata && edata)
            readdata(is);
        else
            throw std::exception();
    }
    ~Formatdata() {
        size_ = 0;
        //Batchprocessed::~Batchprocessed();
    }
    void process() override {
        Batchprocessed::process();
        if (idata && edata) {
            idata->process();
            edata->process();
        }
        //size_ = (idata->size() < edata->size()) ? idata->size() : edata->size();
    };

    virtual void matchiedata() {
        bool p = paused();
        pause();
        std::vector<E> estr {};
        int offset = 0;
        bool match=true;
        if (idata->size() != edata->size())
            for (int m = 0; m < idata->size() && m+offset < edata->size() && match;++m) {
                E e = edata->getdata(m + offset);
                I i = idata->getdata(m);
                match = match && lookup(e) == i;
                if (match)
                    estr.push_back(e);
                else {
                    ++offset;
                }
            }
        edata->readvector(estr);
    }
};

template<typename I, typename E, typename IBPD, typename EBPD>
I Formatdata<I,E,IBPD,EBPD>::lookup(E ve) {
    bool found = false;
    int n = 0;
    int sz = size();
    if (sz <= 0)
        throw std::out_of_range("I lookup: Cannot find if size is zero");
    E e = edata->getdata(0);
    //std::cout << "ve: " << ve << "e: " << e << "\n";
    found = (ve == e);
    while (!found && n < sz-1) {
        n++;
        e = edata->getdata(n);
        //std::cout << "ve: " << ve << "e: " << e << " sz: " << sz << "\n";
        found = (ve == e);
    }
    if (found)
        return idata->getdata(n);
    else
        throw std::out_of_range("I lookup: Unknown data element");
}

template<typename I, typename E, typename IBPD, typename EBPD>
E Formatdata<I,E,IBPD,EBPD>::lookup(I vi) {
    bool found = false;
    int n = 0;
    int sz = size();
    if (sz <= 0)
        throw std::out_of_range("E lookup: Cannot find if size is zero");
    I i = idata->getdata(0);
    found = (vi == i);
    while (!found && n < sz-1) {
        n++;
        i = idata->getdata(n);
        found = (vi == i);
    }
    if (found)
        return edata->getdata(n);
    else
        //throw std::out_of_range("E lookup: Unknown data element");
        return "UNKNWN";
}

#endif //HELLYTOOLCPP_FORMATDATA_CPP
