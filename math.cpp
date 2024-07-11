//
// Created by peterglenn on 7/10/24.
//

#include <iostream>
#include <map>
#include <algorithm>
#include <vector>
#include <cmath>
#include "mathfn.cpp"

inline bool is_number(const std::string& s)
{
    if (!s.empty() && s[0] == '-')
        return (is_number(s.substr(1,s.size()-1)));
    return !s.empty() && std::find_if(s.begin(),
        s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}


enum class logicalconnective {lcand, lcor};

struct logicalsentence {
    int item {};
    std::vector<logicalsentence> ls {};
    logicalconnective lc {logicalconnective::lcand};
    bool negated {false};
};

inline bool evalsentence( logicalsentence ls, std::vector<bool> literals ) {

    bool res;

    if (ls.ls.size()==0) {
        if (ls.item >= 0 && ls.item < literals.size()) {
            res = literals[ls.item];
        } else {
            if (ls.item < 0 && (literals.size() + ls.item >= 0))
                res = literals[literals.size() + ls.item];
            else
                return true;
        }
    }


    if (ls.ls.size() > 0 && ls.lc == logicalconnective::lcand) {
        res = true;
        int n = 0;
        while (res && n < ls.ls.size())
            res = res && evalsentence( ls.ls[n++], literals);
    }
    if (ls.ls.size() > 0 && ls.lc == logicalconnective::lcor) {
        res = false;
        int n = 0;
        while (!res && n < ls.ls.size())
            res = res || evalsentence(ls.ls[n++],literals);
    }
    //std::cout << (ls.ls.size()) << "ls.ls.size()\n";
    //std::cout << res << " (res)\n";
    return ls.negated != res;
}

inline logicalsentence lscombine( const logicalsentence ls1, const logicalsentence ls2, const logicalconnective lc ) {
    logicalsentence res;
    res.ls.push_back(ls1);
    res.ls.push_back(ls2);
    res.lc = lc;
    res.negated = false;
    res.item = 0;
    return res;
}


enum class formulaoperator
{foliteral,fofunction, foconstant, foplus, fominus, fotimes, fodivide, foexponent,
folte, folt, foe,fone,fogte,fogt,
foand,foor,fonot,fotrue,fofalse};

inline std::map<std::string,formulaoperator> operatorsmap
    {{"^",formulaoperator::foexponent},
        {"*",formulaoperator::fotimes},
        {"/",formulaoperator::fodivide},
        {"+",formulaoperator::foplus},
        {"-",formulaoperator::fominus},
        {"AND",formulaoperator::foand},
        {"OR",formulaoperator::foor},
        {"NOT",formulaoperator::fonot},
        {"&&",formulaoperator::foand},
        {"||",formulaoperator::foor},
        {"!",formulaoperator::fonot},
        {"==",formulaoperator::foe},
        {"<=",formulaoperator::folte},
        {"<",formulaoperator::folt},
        {">=",formulaoperator::fogte},
        {">",formulaoperator::fogt},
        {"!=",formulaoperator::fone}};


inline std::vector<std::string> parsecomponents( std::string str) {
    std::string partial {};
    std::vector<std::string> components {};
    bool bracketed = false;
    for (auto ch : str) {
        if (ch == ' ') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            continue;
        }
        if (ch == '(') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back("(");
            continue;
        }
        if (ch == ')') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back(")");
            continue;
        }
        if (ch == '+') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back("+");
            continue;
        }
        if (!bracketed && ch == '-') {
            if (components.size() > 0) {
                bool keyword = false;
                for (auto k : operatorsmap)
                    keyword = keyword || k.first == components[components.size()-1];
                if (!keyword && components[components.size()-1] != "(") {
                    if (partial != "") {
                        components.push_back(partial);
                        partial = "";
                    }
                    components.push_back("-");
                    continue;
                }
            }
        }
        if (ch == '*') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back("*");
            continue;
        }
        if (ch == '/') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back("/");
            continue;
        }
        if (ch == '^') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back("^");
            continue;
        }
        if (ch == ',') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back(",");
            continue;
        }

        if (ch == '=')
        {
            if (partial != "")
            {
                if (partial[partial.size()-1] == '<' || partial[partial.size()-1] == '>' || partial[partial.size()-1] == '=')
                {
                    partial.push_back(ch);
                    if (partial.size() > 2)
                        components.push_back(partial.substr(0,partial.size()-2));
                    components.push_back( partial.substr(partial.size() - 2, 2));
                    partial = "";
                    continue;
                }
            }
        }
        if (ch == '&')
        {
            if (partial != "")
            {
                if (partial[partial.size()-1] == '&')
                {
                    partial.push_back(ch);
                    if (partial.size() > 2)
                        components.push_back(partial.substr(0,partial.size()-2));
                    components.push_back( partial.substr(partial.size() - 2, 2));
                    partial = "";
                    continue;
                }
            }
        }
        if (ch == '|')
        {
            if (partial != "")
            {
                if (partial[partial.size()-1] == '|')
                {
                    partial.push_back(ch);
                    if (partial.size() > 2)
                        components.push_back(partial.substr(0,partial.size()-2));
                    components.push_back( partial.substr(partial.size() - 2, 2));
                    partial = "";
                    continue;
                }
            }
        }

        if (ch == '[') {
            bracketed = true;
        }
        if (ch == ']') {
            bracketed = false;
            partial += ']';
            components.push_back(partial);
            partial = "";
            continue;
        }
        if (ch == '<')
        {
            if (partial != "")
            {
                components.push_back(partial);
            }
            partial = "<";
            continue;
        }
        if (ch == '>')
        {
            if (partial != "")
            {
                components.push_back(partial);
            }
            partial = ">";
            continue;
        }
        if (partial.size() > 0 && partial[partial.size()-1] == '!')
        {
            if (partial.size() > 1)
                components.push_back(partial.substr(0,partial.size()-1));
            components.push_back( partial.substr(partial.size() - 1, 1));
            partial = "";
        }
        if (partial.size() > 0 && (partial[partial.size()-1] == '<' || partial[partial.size()-1] == '>'))
        {
            components.push_back( partial );
            partial = "";
        }
        partial += ch;
    }
    if (partial != "") {
        components.push_back(partial);
    }

    // for (auto c : components)
        // std::cout << "c == " << c << ", ";
    // std::cout << "\n";
    return components;
}


inline logicalsentence parsesentenceinternal (std::vector<std::string> components, int nesting ) {
    std::vector<std::string> left {};
    std::vector<std::string> right = components;
    logicalsentence ls;
    if (components.size() < 1) {
        std::cout << "Error in parsesentenceinternal\n";
        ls.ls.clear();
        ls.item = 0;
        return ls;
    }
    if ((components.size() == 1) && is_number(components[0])) {
        ls.ls.clear();
        ls.item = stoi(components[0]);
        ls.negated = false;
        return ls;
    }


    if (components.size() > 0 && components[0] == "NOT") {
        right.erase(right.begin(), right.begin()+1);
        ls = parsesentenceinternal( right, nesting );
        ls.negated = !ls.negated;
        return ls;
    }



    //for (auto t : components)
    //    std::cout << t << ", ";
    //std::cout << "\n";




    right = components;
    left.clear();
    int j = 0;
    logicalsentence ls1;
    logicalsentence ls2;
    while (j < components.size()) {
        right.erase(right.begin(), right.begin()+1);
        if (nesting <= 0) {
            if ((components[j] == "AND" || components[j] == "OR")) {
                ls1 = parsesentenceinternal(left,nesting);
                ls2 = parsesentenceinternal(right,nesting);
                if (components[j] == "AND") {
                    return lscombine(ls1,ls2,logicalconnective::lcand);
                }
                if (components[j] == "OR") {
                    return lscombine(ls1,ls2,logicalconnective::lcor);
                }
            }
        }
        if (components[j] == "(") {
            ++nesting;
        }
        if (components[j] == ")") {
            --nesting;
        }
        left.push_back(components[j]);
        ++j;
    }

    if (components.size() >= 3) {
        if (components[0] == "(" && components[components.size()-1] == ")") {
            std::vector<std::string>::const_iterator first = components.begin()+1;
            std::vector<std::string>::const_iterator last = components.begin()+components.size()-1;
            std::vector<std::string> tempcomponents(first,last);
            components = tempcomponents;
            return parsesentenceinternal(components,nesting);
        }

    }



    std::cout << "Ill-formed overall sentence\n";
    ls.ls.clear();
    ls.item = 0;
    ls.negated = false;
    return ls;
}

inline logicalsentence parsesentence( std::string sentence ) {
    if (sentence != "")
        return parsesentenceinternal( parsecomponents(sentence), 0 );
    else {
        logicalsentence ls;
        ls.ls.clear();
        ls.item = 0;
        return ls;
    }
}

class formulaclass;

struct fnstruct {
    std::string fn;
    std::vector<formulaclass*> ps;
};

struct formulavalue {
    double val;
    bool bval;
    int lit;
    fnstruct fns;
};

class formulaclass {
public:
    const formulavalue v;
    const formulaclass* fcleft;
    const formulaclass* fcright;
    const formulaoperator fo;
    formulaclass(const formulavalue vin, const formulaclass* fcleftin, const formulaclass* fcrightin, const formulaoperator foin)
        : v{vin}, fcleft(fcleftin), fcright(fcrightin), fo(foin) {}
    ~formulaclass() {
        delete fcleft;
        delete fcright;
        if (fo == formulaoperator::fofunction) {
            for (auto fnf : v.fns.ps)
                delete fnf;
        }
    }
};


inline double evalformula(
    const formulaclass& fc,
    const std::vector<double> literals,
    const std::map<std::string,std::pair<double (*)(std::vector<double>&),int>>* fnptrs = &global_fnptrs ) {
    double res;
    if (fc.fo == formulaoperator::fotrue)
        return true;
    if (fc.fo == formulaoperator::fofalse)
        return false;
    if (fc.fo == formulaoperator::foconstant)
        return fc.v.val;

    if (fc.fo == formulaoperator::foliteral || (fc.fcleft == nullptr && fc.fcright==nullptr)) {
        if (fc.v.lit >= 0 && fc.v.lit < literals.size()) {
            res = literals[fc.v.lit];
        } else {
            if (fc.v.lit < 0 && ((int)literals.size() + fc.v.lit >= 0))
                res = literals[literals.size() + fc.v.lit];
            else {
                std::cout << "Error eval'ing formula\n";
                return 0;
            }
        }
    }

    if (fc.fo == formulaoperator::fofunction) {
        bool found = false;
        double (*fn)(std::vector<double>&);
        for (auto fnptr : *fnptrs) {
            if ( fnptr.first == fc.v.fns.fn ) {
                found = true;
                fn = fnptr.second.first;
            }
        }
        if (!found) {
            std::cout << "Unknown function named " << fc.v.fns.fn << "\n";
            return -1;
        }
        std::vector<double> ps;
        for (auto f : fc.v.fns.ps) {
            ps.push_back(evalformula(*f,literals,fnptrs));
        }
        return fn(ps);
    }

    if (fc.fo == formulaoperator::foplus) {
        res = evalformula( *fc.fcleft, literals,fnptrs ) + evalformula( *fc.fcright, literals,fnptrs );
    }
    if (fc.fo == formulaoperator::fominus) {
        res = evalformula( *fc.fcleft, literals, fnptrs ) - evalformula( *fc.fcright, literals, fnptrs );
    }
    if (fc.fo == formulaoperator::fotimes) {
        res = evalformula( *fc.fcleft, literals, fnptrs ) * evalformula( *fc.fcright, literals, fnptrs );
    }
    if (fc.fo == formulaoperator::fodivide) {
        res = evalformula( *fc.fcleft, literals,fnptrs ) / evalformula( *fc.fcright, literals,fnptrs );
    }
    if (fc.fo == formulaoperator::foexponent) {
        res = pow(evalformula( *fc.fcleft, literals,fnptrs ), evalformula( *fc.fcright, literals,fnptrs ));
    }

    if (fc.fo == formulaoperator::foand) {
        res = evalformula( *fc.fcleft, literals,fnptrs ) && evalformula( *fc.fcright, literals,fnptrs );
    }
    if (fc.fo == formulaoperator::foor) {
        res = evalformula( *fc.fcleft, literals,fnptrs ) || evalformula( *fc.fcright, literals,fnptrs );
    }
    if (fc.fo == formulaoperator::fonot) {
        res = !evalformula( *fc.fcright, literals,fnptrs );
    }

    if (fc.fo == formulaoperator::foe) {
        res = evalformula( *fc.fcleft, literals,fnptrs ) == evalformula( *fc.fcright, literals,fnptrs );
    }
    if (fc.fo == formulaoperator::folte) {
        res = evalformula( *fc.fcleft, literals,fnptrs ) <= evalformula( *fc.fcright, literals,fnptrs );
    }
    if (fc.fo == formulaoperator::folt) {
        res = evalformula( *fc.fcleft, literals,fnptrs ) < evalformula( *fc.fcright, literals,fnptrs );
    }
    if (fc.fo == formulaoperator::fogte) {
        res = evalformula( *fc.fcleft, literals,fnptrs ) >= evalformula( *fc.fcright, literals,fnptrs );
    }
    if (fc.fo == formulaoperator::fogt) {
        res = evalformula( *fc.fcleft, literals,fnptrs ) > evalformula( *fc.fcright, literals,fnptrs );
    }
    if (fc.fo == formulaoperator::fone) {
        res = evalformula( *fc.fcleft, literals,fnptrs ) != evalformula( *fc.fcright, literals,fnptrs );
    }

    return res;
}



inline formulaclass* fccombine( const formulavalue& item, const formulaclass* fc1, const formulaclass* fc2, const formulaoperator fo ) {
    auto res = new formulaclass(item,fc1,fc2,fo);
    return res;
}


inline std::map<formulaoperator,int> precedencemap {
                            {formulaoperator::foexponent,0},
                            {formulaoperator::fotimes,1},
                            {formulaoperator::fodivide,1},
                            {formulaoperator::foplus,2},
                            {formulaoperator::fominus,2},
                            {formulaoperator::foe,3},
                            {formulaoperator::folte,3},
                            {formulaoperator::folt,3},
                            {formulaoperator::fogte,3},
                            {formulaoperator::fogt,3},
                            {formulaoperator::fone,3},
                            {formulaoperator::fonot,4},
                            {formulaoperator::foand,5},
                            {formulaoperator::foor,5}};



inline bool is_operator( const std::string& tok ) {
    for (auto o : operatorsmap)
        if (o.first == tok)
            return true;
    return false;
}

inline formulaoperator lookupoperator( const std::string& tok ) {

    for (auto o : operatorsmap ) {
        if (tok == o.first)
            return o.second;
    }
    return formulaoperator::fotimes;
}

inline bool is_truth( const std::string& tok )
{
    return tok == "TRUE" || tok == "FALSE";
}

inline formulaoperator lookuptruth( const std::string& tok )
{
    return tok == "TRUE" ? formulaoperator::fotrue : formulaoperator::fofalse;
}

inline int operatorprecedence( const std::string& tok1, const std::string& tok2) {
    formulaoperator tok1o = lookupoperator(tok1);
    formulaoperator tok2o = lookupoperator(tok2);

    if (precedencemap[tok1o] == precedencemap[tok2o])
        return 0;
    if (precedencemap[tok1o] < precedencemap[tok2o])
        return 1;
    if (precedencemap[tok1o] > precedencemap[tok2o])
        return -1;
}

inline bool is_function(std::string tok) {
    bool res = !tok.empty();
    for (auto c : tok)
        res = res && isalpha(c);
    return res;
}

inline bool is_literal(std::string tok) {
    return tok.size() > 2 && tok[0] == '[' && tok[tok.size()-1] == ']';
}

inline int get_literal(std::string tok) {
    if (is_literal(tok))
        return stoi(tok.substr(1,tok.size()-2));
    return 0;
}

inline std::vector<std::string> Shuntingyardalg( std::vector<std::string> components) {
    std::vector<std::string> output {};
    std::vector<std::string> operatorstack {};

    int n = 0;
    while( n < components.size()) {
        const std::string tok = components[n++];
        if (is_number(tok)) {
            output.push_back(tok);
            continue;
        }
        if (is_literal(tok)) {
            output.push_back(tok);
            continue;
        }
        if (is_truth(tok)) {
            output.push_back(tok);
            continue;
        }
        if (is_operator(tok)) {
            if (operatorstack.size() >= 1) {
                std::string ostok = operatorstack[operatorstack.size()-1];
                while (is_operator(ostok) && ostok != "(" && operatorprecedence(ostok,tok) >= 0) {
                    output.push_back(ostok);
                    operatorstack.resize(operatorstack.size()-1);
                    if (operatorstack.size() >= 1)
                        ostok = operatorstack[operatorstack.size()-1];
                    else
                        break;
                }
            }
            operatorstack.push_back(tok);
            continue;
        }
        if (is_function(tok)) {
            operatorstack.push_back(tok);
            continue;
        }
        if (tok == ",") {
            if (operatorstack.size() >= 1) {
                std::string ostok = operatorstack[operatorstack.size()-1];
                while (ostok != "(") {
                    output.push_back(ostok);
                    operatorstack.resize(operatorstack.size()-1);
                    ostok = operatorstack[operatorstack.size()-1];
                }
            }
            continue;
        }
        if (tok == "(") {
            operatorstack.push_back(tok);
        }
        if (tok == ")") {
            if (operatorstack.size() >= 1) {
                std::string ostok = operatorstack[operatorstack.size()-1];
                while (ostok != "(") {
                    if (operatorstack.empty()) {
                        std::cout << "Error mismatched parentheses\n";
                        return output;
                    }
                    output.push_back(ostok);
                    operatorstack.resize(operatorstack.size()-1);
                    ostok = operatorstack[operatorstack.size()-1];
                }
                if (operatorstack.empty() || operatorstack[operatorstack.size()-1] != "(") {
                    std::cout << "Error mistmatched parentheses\n";
                    return output;
                }
                operatorstack.resize(operatorstack.size()-1);
                if (operatorstack.size() > 0) {
                    ostok = operatorstack[operatorstack.size()-1];
                    output.push_back(ostok);
                    operatorstack.resize(operatorstack.size()-1);
                }
            }
            continue;
        }
    }
    while (operatorstack.size()> 0) {
        std::string ostok = operatorstack[operatorstack.size()-1];
        if (ostok == "(" || ostok == ")") {
            std::cout << "Error mismatched parentheses\n";
            return output;
        }
        output.push_back(ostok);
        operatorstack.resize(operatorstack.size()-1);
    }

    // for (auto o : output)
        // std::cout << o << ", ";
    // std::cout << "\n";

    return output;

}


inline formulaclass* parseformulainternal(
    const std::vector<std::string>& q,
    int& pos,
    const std::map<std::string,std::pair<double (*)(std::vector<double>&),int>>* fnptrs = &global_fnptrs )
{
    if (pos == -1)
        pos = q.size();
    while( pos > 0) {
        --pos;
        std::string tok = q[pos];
        if (is_operator(tok))
        {
            formulaoperator o = lookupoperator(tok);
            formulaclass* fcright = parseformulainternal(q,pos,fnptrs);
            formulaclass* fcleft = nullptr;
            if (o != formulaoperator::fonot)
            {
                fcleft = parseformulainternal(q,pos,fnptrs);
            }
            if (fcright)
                return fccombine({0},fcleft,fcright,o);
        }
        if (is_function(tok)) {
            std::vector<formulaclass*> ps {};
            // double (*f2)(std::vector<double>&);
            int argcnt = 0;
            for (auto f : *fnptrs)
                if (f.first == tok)
                    argcnt = f.second.second;
            std::vector<formulaclass*> psrev {};
            for (int i = 0; i < argcnt; ++i) {
                psrev.push_back(parseformulainternal(q,pos,fnptrs));
            }
            for (int i = psrev.size()-1; i >= 0; --i)
                ps.push_back(psrev[i]);

            formulavalue fv {};
            fv.fns.fn = tok;
            fv.fns.ps = ps;
            return fccombine(fv,nullptr,nullptr,formulaoperator::fofunction);
        }
        if (is_literal(tok)) {
            formulavalue fv {};
            fv.lit = get_literal(tok);
            return fccombine(fv,nullptr,nullptr,formulaoperator::foliteral);
        }
        if (is_number(tok)) {
            formulavalue fv {};
            fv.val = stoi(tok);
            return fccombine(fv,nullptr,nullptr,formulaoperator::foconstant);
        }
        if (is_truth(tok))
        {
            formulavalue fv {};
            formulaoperator t = lookuptruth(tok);
            fv.bval = t == formulaoperator::fotrue;
            return fccombine(fv,nullptr,nullptr,t);
        }
    }
    std::cout << "Error in parsing formula \n";
    auto fc = new formulaclass({0},nullptr,nullptr,formulaoperator::foconstant);
    return fc;
}

inline formulaclass* parseformula(
    const std::string sentence,
    const std::map<std::string,std::pair<double (*)(std::vector<double>&),int>>* fnptrs = &global_fnptrs  )
{
    if (sentence != "") {
        std::vector<std::string> components = Shuntingyardalg(parsecomponents(sentence));
        int pos = components.size();
        return parseformulainternal( components,pos,fnptrs);
    } else {
        auto fc = new formulaclass({0},nullptr,nullptr,formulaoperator::foconstant);
        return fc;
    }
}


/*
inline void parseequation( std::string* equation, std::string* lhs, std::string* rhs, int* eqtype) {

    int pos0 = equation->find ("==");
    int pos1 = equation->find(">=");
    int pos2 = equation->find( ">");
    int posn1 = equation->find( "<=");
    int posn2 = equation->find( "<");

    int pos;

    if (pos0 != std::string::npos) {
        pos = pos0;
        *eqtype = 0;
    }
    if (pos1 != std::string::npos) {
        pos = pos1;
        *eqtype = 1;
    }
    if (pos2 != std::string::npos && pos1 == std::string::npos) {
        pos = pos2;
        *eqtype = 2;
    }
    if (posn1 != std::string::npos) {
        pos = posn1;
        *eqtype = -1;
    }
    if (posn2 != std::string::npos && posn1 == std::string::npos) {
        pos = posn2;
        *eqtype = -2;
    }
    *lhs = equation->substr(0,pos-1);
    *rhs = equation->substr(pos+2,equation->size()-pos-2);
    //std::cout << *lhs << " === " << *rhs << " , " << *eqtype << "\n";

}
*/
