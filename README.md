FlagCalc is intended to assist the theoretical mathematician at all levels of profession and expertise, including those in secondary school settings and those doing the most cutting edge research.

It arose from a simple curiosity around asymptotic extremal graph theory, namely to decipher the first pages of research and books on graph theory. Initially, that meant curiosity explored around computing isomorphisms between any two similarly-dimensioned graphs, which led to automorphism counts and a “fingerprinting” of any graph, done automatically, and leading to one suggestion of how one can order a set of graphs linearly. The early goal was simply a database of graphs unique up to isomorphism.

Along the way the tool became a veritable workhorse in applied Human-Computer Interface: how to use both the inheritance and class methods of C++ best design practices, with bit-level machine code-almost algorithms. Indeed, it is a point of fascination to the tool’s author, how fast it in fact can be on a multi-threaded CPU. But along with that is the interest in making the tool livable, usable, “smart”, surprising in its clever connections that it helps us to make, etc.

Therefore, to get down to it: the tool uses a long command-line interface to make requests, to put things on the “workspace” stack (mostly graphs, but also results from computations, etc.), to produce graphs either with an abstracted choice from amongst your favorite “random graph algorithms”, or by reading in from a file human-made (using a friendly set of conventions for inputting graphs) or say a Python-created text file that may for instance be enumerating all graphs on a certain dimension.

To start, therefore, here are ten standard invocations of “flagcalc”, with explanation:

0. First,

flagcalc -h 

outputs a help screen, automated by computer code such that any added-on graph random algorithms or command-line features or criteria and measures identify themselves automatically. However, please continue to read this README!

1. The absolute standard for any low-count work (under say ten thousand graphs on my Intel i7 desktop CPU) is to invoke with the “-f all” command: “please fingerprint and sort into equivalence classes all the graphs found on the workplace. Therefore, to step back, we need to populate the workplace first (and here is something a bit different from customary command-line Linux convention: order DOES MATTER: so we invoke as follows:

flagcalc -d testgraph23.dat testbip12.dat -f all

This reads all the graphs from the two files named and places them on the workplace stack. Then it fingerprints them all and lists the equivalence classes. Which brings us to 2:

2. Isomorphisms and automorphisms

flagcalc -d <your filenames here> -f all -i sorted

The “-i” means compute an isomorphism count, if left by itself then “-i” simply lists the number of isomporhisms between the last two graphs added to the stack, or the automorphism count if only one graph is on the stack. But! As used above, it computes the automorphism count for one graph from each fingerprint-equivalence class (as of this writing, seen in practice to be equivalent to isomorphism-equivalence class). If it is too slow to compute a fingerprint for all (“-f all”) then “-i all” can be used instead; if “-i sorted” returns empty, that may be because no “-f all” preceded it on the command-line.

3. Verbosity: this is really the place to begin, but we wanted some tantalizing things to be verbose about first. Verbosity was an idea born in long sessions in Python and C++ trying to make human-readable output, not too lengthy and not too short. For example, we don’t need “-f all” to print literally each fingerprint out, unless we are focusing on that in particular. Also, while “-i sorted” or “-i all” does produce say three million automorphisms in a few seconds, it need not generally list each and every map. Just the count will suffice. For the sake of brevity: use

flagcalc <other options> -v i=minimal2.cfg min

to read in from minimal2.cfg a list of tokens, and then add to that list one more token, “min” (by omitting this last “min” one will get an adjacency matrix and other info for each graph found on the workplace stack: useful for working with a couple dozen graphs, but too much for random samples and tests on several thousand graphs).

In other words, order matters: put “-v” LAST on the command line, or repeat it in a couple places (I’ve seen command lines of useful commands numbering in the dozens, and more theoretically possible). You can for example just output the number of automorphisms by specifying “Iso” token, then “Noiso” to exclude the actual maps themselves. See the source file “workspace.h” for a current list of verbosity tokens, and feel free to program in your own as needed, if you are so inclined.

4. Populating instead of from .dat files, with random graphs

flagcalc -r 10 12 1000

will produce one thousand random graphs on ten vertices and an average of twelve edges. There are presently five random algorithms to choose from: “-r 10 12 1000 r1” through “-r 10 12 1000 r5”, and one can easily add more, including newly “parameterized” random algorithms, such as “-r 10 12 1000 r6(50)”.

5. Checking likelihood of two random graphs being isomorphic

flagcalc -R 10 12 10000

will sample two graphs with the optional listed random algorithm “r1...r5” (default is r1) and count if those two are isomorphic

6. The heart of the present uses of the tool, and coding work ongoing, is towards “-a”: compute the likelihood over say twenty-thousand graphs (random or manually entered, or automatically populated by some computer script) having any number of named criteria, and then amongst those that pass the criterion, compute the named “measure” of the graph, e.g. “girth”, “average edge count”, “diameter”, etc. I use terminology from the opening pages of Diestel’s 2017 graduate textbook “Graph Theory”.

flagcalc <your commands here> -a connc girthm all

will use criterion “connected criterion” and for those that are connected graphs, will apply “girth measure”. Note “all” is required, unless one has sorted previously using “-f all”, in which case “sorted” is possible in place of “all”; in the absence of either, it will just apply the criteria and measures to the last graph on the workplace stack.

7. Continuing,

flagcalc <your commands here> -a connc treec radiusm s=”(NOT 0) AND 1” all

will test “connected criterion” and “tree criterion”, accepting the logical “NOT” of the first and the positive truth of the second, I.e all forests (and here is the code I personally just used to confirm that “tree criterion” still departs from convention by including “forests”: 

“flagcalc -r 10 12 1000 -a connc treec l=AND all -v i=minimal2.cfg min”

8. Referring to the tool’s name, one can input a test graph to induce an embedding in the graphs on the workplace, using

flagcalc <your commands here> -a f=”-abcda” all 

will test for the cycle “abcda”, being embeddable such that adjacency and non-adjacency are preserved. Likewise “if=<filename>” inputs such “flags” from a file.

9. Lastly, the cousin to verbosity is “machine-readable output”: “-g” outputs the graphs on the workplace stack to a file, in machine-readable format (so one can turn right around and recreate the workplace last used by a call to “-d <filename>” at the start of the next command line invocation.

flagcalc <your commands here> -g o=<filename> overwrite passed

where one chooses between “overwrite” and “append”, and uses optionally “passed” or “passed=2” to output only those that passed the second criterion, or “passed=-1” to output those that passed the last criterion. All these can be interspersed, with say two invocations of a “-a” to get clock speed results on comparative algorithms, and “-g” used multiple times, once to output all graphs and once just those that passed the criteria.

That is a very brief summary of what the tool does. Finally, here we explain the formalism for inputting graphs:

1. graph vertices are labelled by convention with an alphabetic character followed by any number of numerals, colons, underscores. When this isn’t sufficient (we have “a...z” and “A...Z” giving fifty-two vertices, already before even using the numerals option) we can use anything as a name provided we enclose it in curly braces, and that it doesn’t use the special characters “=” and “+” and “-” and “!”.

2. any jumble of vertex labels immediately adjacent to each other produces a clique or complete set of those vertices all connected to every other possible vertex in that list (this default behavior is a relic from Helly Tool, where we are speedily trying to input complete set edge coverings). E.g.

abcde 

or

*abcde

each do the same thing, making a K_5, with edges ab ac ad ae bc bd be cd ce de.

3. A leading “-” means “path with the following sequence of vertices” (including cycles, if the path ends with the vertex it began on.

-abcde 

produces edges ab bc cd de

4. An interposed “=”, or n interposed “=” produce a n-partite graph with each section one component:

abcd=efgh

is a bipartite graph on eight vertices, and

ab=cde=fghij

is a tripartite graph.

5. An interposed “+” is a radial graph, such that

a+bcdefg

is a star with center a and edges ab ac ad ae af ag

likewise
abcd+efg

is a K_4 at the center, with edges ae be ce de af bf cf df ce cf cg de df dg.

6. Finally, the “!” does the NOT of whatever immediately follows

!abcde is five outlier vertices with no edges (it being the inverse of the complete graph K_5).

Each of these six are generous to existing structure: if one does a bipartite graph on abcd=efgh, it will not by default erase existing edges between say a and b. If one wishes to erase, use as just illustrated the !abcde. Finally, one will see earlier iterations of this input format that required naming all the vertices then ending that section with “END”. Here, to name say a b c is fine; each is a little K_1. And here, too, for backwards compatibility, after describing one graph, if one wishes to add a second or third, etc., graph to the file, punctuate each with two “END”s: “END END”.

Please have fun and thank you for your time.
