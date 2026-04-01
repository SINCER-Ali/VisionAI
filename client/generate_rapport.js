const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, LevelFormat,
        HeadingLevel, BorderStyle, WidthType, ShadingType,
        PageNumber, PageBreak, TabStopType, TabStopPosition } = require('docx');
const fs = require('fs');

const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

function p(text, opts = {}) {
    const runs = [];
    if (typeof text === 'string') {
        runs.push(new TextRun({ text, size: opts.size || 24, font: "Calibri", bold: opts.bold, italics: opts.italics, color: opts.color }));
    } else if (Array.isArray(text)) {
        text.forEach(t => {
            if (typeof t === 'string') runs.push(new TextRun({ text: t, size: opts.size || 24, font: "Calibri" }));
            else runs.push(new TextRun({ size: 24, font: "Calibri", ...t }));
        });
    }
    return new Paragraph({
        children: runs,
        spacing: { after: opts.after !== undefined ? opts.after : 200, before: opts.before || 0, line: opts.line || 276 },
        alignment: opts.align || AlignmentType.JUSTIFIED,
        indent: opts.indent ? { firstLine: 360 } : undefined,
        pageBreakBefore: opts.pageBreak || false,
    });
}

function heading(text, level, pageBreak = false) {
    return new Paragraph({
        heading: level,
        children: [new TextRun({ text, size: level === HeadingLevel.HEADING_1 ? 36 : level === HeadingLevel.HEADING_2 ? 30 : 26, bold: true, font: "Calibri", color: level === HeadingLevel.HEADING_1 ? "1F3864" : level === HeadingLevel.HEADING_2 ? "2E75B6" : "404040" })],
        spacing: { before: level === HeadingLevel.HEADING_1 ? 400 : 300, after: 200 },
        pageBreakBefore: pageBreak,
    });
}

function cell(text, opts = {}) {
    return new TableCell({
        borders,
        width: { size: opts.width || 2340, type: WidthType.DXA },
        margins: cellMargins,
        shading: opts.header ? { fill: "1F3864", type: ShadingType.CLEAR } : undefined,
        children: [new Paragraph({ children: [new TextRun({ text, size: 22, font: "Calibri", bold: opts.header, color: opts.header ? "FFFFFF" : "333333" })], alignment: opts.align || AlignmentType.LEFT })],
    });
}

const children = [];

// ═══ PAGE DE GARDE ═══
children.push(new Paragraph({ spacing: { before: 3000 } }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: "ESGI - Ecole Superieure de Genie Informatique", size: 28, font: "Calibri", color: "666666" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: "Projet Annuel - Intelligence Artificielle & Machine Learning", size: 24, font: "Calibri", color: "888888" })] }));
children.push(new Paragraph({ spacing: { before: 600 } }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: "VisionAI", size: 56, bold: true, font: "Calibri", color: "1F3864" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: "Framework de Machine Learning en Rust", size: 32, font: "Calibri", color: "2E75B6" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 50 }, children: [new TextRun({ text: "avec bindings Python, API REST et client web", size: 26, font: "Calibri", color: "666666", italics: true })] }));
children.push(new Paragraph({ spacing: { before: 800 } }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: "Livrable intermediaire - Semaine du 6 avril 2026", size: 24, font: "Calibri", color: "CC6600", bold: true })] }));
children.push(new Paragraph({ spacing: { before: 400 } }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Valentin Brouchoud", size: 26, font: "Calibri", bold: true })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Ali Sincer", size: 26, font: "Calibri", bold: true })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Thinina (Nina)", size: 26, font: "Calibri", bold: true })] }));
children.push(new Paragraph({ spacing: { before: 600 } }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Annee 2025-2026", size: 26, font: "Calibri", color: "888888" })] }));

// ═══ SOMMAIRE ═══
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(heading("Sommaire", HeadingLevel.HEADING_1));
const sommaire = [
    "1. Introduction",
    "2. Le projet : ce qu'on doit faire et ou on en est",
    "3. Pourquoi on a choisi Rust (et ce que ca implique)",
    "4. Architecture du projet",
    "5. Apprendre l'algebre lineaire : Vector et Matrix",
    "6. Comprendre les fonctions d'activation",
    "7. Le modele lineaire : notre premier pas en ML",
    "8. Le MLP : la ou ca devient serieux",
    "9. La backpropagation : ce qu'on a appris en la codant",
    "10. Sauvegarder un modele : la serialisation",
    "11. L'API REST : rendre le modele accessible",
    "12. Les bindings Python : connecter Rust et Python",
    "13. L'application cliente web",
    "14. Tests : comment on verifie que ca marche",
    "15. Nos resultats (et ce qu'on en comprend)",
    "16. Ce qui nous a pose probleme",
    "17. Etat d'avancement et prochaines etapes",
];
sommaire.forEach(title => {
    children.push(new Paragraph({
        children: [new TextRun({ text: title, size: 24, font: "Calibri" })],
        spacing: { after: 80 },
    }));
});

// ═══ 1. INTRODUCTION ═══
children.push(heading("1. Introduction", HeadingLevel.HEADING_1, true));

children.push(p("Ce rapport presente l'avancement de notre projet annuel VisionAI. On dit bien avancement, parce que le projet n'est pas termine : c'est un livrable intermediaire pour la semaine du 6 avril, et il reste encore du travail pour la soutenance finale.", { indent: true }));

children.push(p("L'objectif du projet, c'est de construire un framework de Machine Learning en Rust. Avant de commencer, aucun de nous trois n'avait fait de ML autrement qu'en appelant des fonctions toutes faites en Python (genre model.fit() sur scikit-learn). On ne savait pas vraiment ce qui se passait derriere. Et pour Rust, on partait quasiment de zero aussi.", { indent: true }));

children.push(p("Du coup, ce rapport va autant parler de ce qu'on a construit que de ce qu'on a appris en le construisant. On va essayer d'expliquer les concepts au fur et a mesure, parce que c'est justement le but du projet : comprendre comment ca marche, pas juste faire tourner un truc.", { indent: true }));

children.push(p("On est trois : Valentin s'occupe du core ML (les maths, les modeles, les tests), Ali gere la partie donnees et experimentations en Python, et Nina a fait l'API serveur et les bindings Python. On detaillera la repartition plus loin.", { indent: true }));

// ═══ 2. LE PROJET ═══
children.push(heading("2. Le projet : ce qu'on doit faire et ou on en est", HeadingLevel.HEADING_1, true));

children.push(p("Le cahier des charges nous demande de creer une bibliotheque de Machine Learning dans un langage systeme (C, C++ ou Rust), avec plusieurs interfaces pour l'utiliser. Concretement, on doit livrer :", { indent: true }));

children.push(p([{ text: "Un core ML ", bold: true }, { text: "en Rust avec les maths de base et au moins deux types de modeles (lineaire + reseau de neurones)." }]));
children.push(p([{ text: "Des bindings Python ", bold: true }, { text: "pour pouvoir utiliser nos modeles depuis des notebooks Jupyter." }]));
children.push(p([{ text: "Un serveur API REST ", bold: true }, { text: "pour deployer un modele et faire des predictions a distance." }]));
children.push(p([{ text: "Une application cliente ", bold: true }, { text: "pour tester visuellement le tout (uploader une image et voir le resultat)." }]));
children.push(p([{ text: "Des tests ", bold: true }, { text: "qui prouvent que les modeles apprennent correctement." }]));
children.push(p([{ text: "Un rapport ", bold: true }, { text: "d'au moins 20 pages et une soutenance." }]));

children.push(heading("2.1. Ou on en est aujourd'hui", HeadingLevel.HEADING_2));

const statusTable = new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [4000, 1800, 3560],
    rows: [
        new TableRow({ children: [
            cell("Element", { width: 4000, header: true }),
            cell("Statut", { width: 1800, header: true }),
            cell("Commentaire", { width: 3560, header: true }),
        ]}),
        new TableRow({ children: [ cell("Core ML (maths, modeles)", { width: 4000 }), cell("Fait", { width: 1800 }), cell("Vector, Matrix, activations, MLP", { width: 3560 }) ]}),
        new TableRow({ children: [ cell("Serialisation des modeles", { width: 4000 }), cell("Fait", { width: 1800 }), cell("JSON + binaire", { width: 3560 }) ]}),
        new TableRow({ children: [ cell("Tests (40 tests)", { width: 4000 }), cell("Fait", { width: 1800 }), cell("XOR, AND, OR, multi-classes, etc.", { width: 3560 }) ]}),
        new TableRow({ children: [ cell("API REST", { width: 4000 }), cell("Fait", { width: 1800 }), cell("/predict, /train, /models", { width: 3560 }) ]}),
        new TableRow({ children: [ cell("Bindings Python", { width: 4000 }), cell("Fait", { width: 1800 }), cell("PyO3 + maturin", { width: 3560 }) ]}),
        new TableRow({ children: [ cell("Client web", { width: 4000 }), cell("Fait", { width: 1800 }), cell("Upload image + prediction", { width: 3560 }) ]}),
        new TableRow({ children: [ cell("Dataset reel (images)", { width: 4000 }), cell("A faire", { width: 1800 }), cell("Script pret, donnees a telecharger", { width: 3560 }) ]}),
        new TableRow({ children: [ cell("Notebooks d'experimentation", { width: 4000 }), cell("A faire", { width: 1800 }), cell("Courbes, matrices de confusion", { width: 3560 }) ]}),
        new TableRow({ children: [ cell("Analyse des phenomenes ML", { width: 4000 }), cell("A faire", { width: 1800 }), cell("Sur/sous-apprentissage, hyperparametres", { width: 3560 }) ]}),
        new TableRow({ children: [ cell("Soutenance", { width: 4000 }), cell("A preparer", { width: 1800 }), cell("", { width: 3560 }) ]}),
    ],
});
children.push(statusTable);
children.push(p(""));

children.push(p("Comme on peut le voir, la partie technique (le code) est bien avancee, mais il nous reste du travail sur la partie experimentation et analyse. C'est prevu pour la semaine prochaine.", { indent: true }));

// ═══ 3. POURQUOI RUST ═══
children.push(heading("3. Pourquoi on a choisi Rust (et ce que ca implique)", HeadingLevel.HEADING_1, true));

children.push(p("Aucun de nous ne connaissait vraiment Rust avant ce projet. On avait fait du C en premiere annee et du Python depuis, mais Rust c'etait nouveau. On l'a choisi parce que le prof l'avait mentionne comme une option interessante et qu'on voulait apprendre quelque chose de different.", { indent: true }));

children.push(heading("3.1. Ce qu'on a decouvert en apprenant Rust", HeadingLevel.HEADING_2));

children.push(p("Rust a une particularite qui le rend tres different des langages qu'on connaissait : il verifie a la compilation que le programme ne fait pas d'erreurs de memoire. En Python, si on fait une erreur, le programme plante au moment ou on le lance. En C, il peut meme ne pas planter mais faire n'importe quoi (un bug silencieux). En Rust, le compilateur refuse de compiler si il detecte un probleme potentiel.", { indent: true }));

children.push(p([{ text: "Le borrow checker. ", bold: true }, { text: "C'est le nom du systeme de Rust qui verifie la memoire. En gros, en Rust, chaque donnee a un seul \"proprietaire\" a la fois. Si une fonction veut lire une donnee, elle l'\"emprunte\" temporairement. Et Rust interdit d'avoir quelqu'un qui modifie une donnee pendant que quelqu'un d'autre est en train de la lire. Au debut, on trouvait ca super penible parce que le compilateur nous bloquait tout le temps. Mais on a compris que ca nous empechait de faire des bugs qu'on aurait mis des heures a trouver autrement." }]));

children.push(p([{ text: "Pas de garbage collector. ", bold: true }, { text: "En Python ou Java, il y a un programme en arriere-plan (le \"ramasse-miettes\" ou garbage collector) qui surveille la memoire et libere automatiquement ce dont on n'a plus besoin. C'est pratique, mais ca ralentit le programme parce que ce nettoyage prend du temps. Rust n'en a pas : il calcule a la compilation quand liberer la memoire, sans aucun cout au moment de l'execution. C'est pour ca que Rust est aussi rapide que le C." }]));

children.push(p([{ text: "Cargo. ", bold: true }, { text: "C'est l'outil qui gere la compilation et les dependances en Rust (un peu comme pip en Python, mais en mieux). On ecrit les dependances dans un fichier Cargo.toml, on tape cargo build, et tout se telecharge et se compile. Comparee a l'enfer des Makefiles en C, c'est le jour et la nuit." }]));

children.push(heading("3.2. Pourquoi c'est adapte au ML", HeadingLevel.HEADING_2));

children.push(p("Le Machine Learning, c'est beaucoup de calcul. Quand on entraine un reseau de neurones, on fait des milliers de multiplications de matrices en boucle. En Python pur, ca serait extremement lent. C'est pour ca que les bibliotheques ML comme TensorFlow ou PyTorch sont en fait ecrites en C++ sous le capot : Python sert juste d'interface.", { indent: true }));

children.push(p("Avec Rust, on est directement au niveau du C++ en termes de vitesse, sans les risques de bugs memoire. Et grace a PyO3 (qu'on verra plus loin), on peut quand meme utiliser nos modeles depuis Python pour la partie experimentation.", { indent: true }));

children.push(heading("3.3. Ce qui a ete difficile", HeadingLevel.HEADING_2));

children.push(p("On ne va pas mentir : apprendre Rust en meme temps que le ML, c'etait ambitieux. Les premiers jours, on passait plus de temps a se battre avec le compilateur qu'a coder. Le borrow checker refusait notre code de backpropagation parce qu'on essayait de lire et modifier le meme tableau en meme temps. On a du repenser notre facon de structurer le code, ce qui au final l'a rendu plus propre.", { indent: true }));

// ═══ 4. ARCHITECTURE ═══
children.push(heading("4. Architecture du projet", HeadingLevel.HEADING_1, true));

children.push(p("Le projet est decoupe en trois parties independantes qu'on appelle des \"crates\" en Rust (l'equivalent d'un package Python ou d'une bibliotheque). Chaque crate a son propre dossier et son propre fichier de configuration Cargo.toml.", { indent: true }));

const archTable = new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2200, 3200, 3960],
    rows: [
        new TableRow({ children: [
            cell("Crate", { width: 2200, header: true }),
            cell("Role", { width: 3200, header: true }),
            cell("Ce qu'on y trouve", { width: 3960, header: true }),
        ]}),
        new TableRow({ children: [
            cell("core_lib", { width: 2200 }),
            cell("Le coeur : toutes les maths et les modeles ML", { width: 3200 }),
            cell("Vector, Matrix, activations, modele lineaire, MLP, serialisation", { width: 3960 }),
        ]}),
        new TableRow({ children: [
            cell("api_server", { width: 2200 }),
            cell("Un serveur web pour utiliser les modeles a distance", { width: 3200 }),
            cell("Endpoints /predict, /train, /models (framework Axum)", { width: 3960 }),
        ]}),
        new TableRow({ children: [
            cell("python_binding", { width: 2200 }),
            cell("Permet d'appeler le code Rust depuis Python", { width: 3200 }),
            cell("Classes PyVector, PyMLP, fonctions train/predict (PyO3)", { width: 3960 }),
        ]}),
    ],
});
children.push(archTable);
children.push(p(""));

children.push(p("En plus de ces trois crates, on a un dossier client/ pour l'application web et un dossier notebooks/ pour les scripts Python d'experimentation. Le tout est regroupe dans un \"workspace\" Cargo, ce qui veut dire qu'on peut tout compiler d'un coup avec une seule commande.", { indent: true }));

children.push(p("Cette organisation, on ne l'a pas trouvee du premier coup. Au debut, on avait tout dans un seul dossier, et c'etait vite le bazar. C'est en lisant la doc Rust sur les workspaces qu'on a compris comment bien decouper le projet.", { indent: true }));

// ═══ 5. ALGEBRE LINEAIRE ═══
children.push(heading("5. Apprendre l'algebre lineaire : Vector et Matrix", HeadingLevel.HEADING_1, true));

children.push(p("Avant ce projet, les vecteurs et les matrices c'etait un truc abstrait qu'on avait vu en cours de maths. On savait vaguement ce que c'etait, mais on ne les avait jamais implementes nous-memes. Et pourtant, c'est la base de tout en ML : les donnees sont des vecteurs, les poids du modele sont des matrices, l'entrainement c'est des multiplications de matrices en boucle.", { indent: true }));

children.push(heading("5.1. Le type Vector", HeadingLevel.HEADING_2));

children.push(p("Notre Vector, c'est un tableau de nombres decimaux (des f64, c'est-a-dire des nombres a virgule stockes sur 64 bits) avec des operations mathematiques dessus. Par exemple, le produit scalaire (\"dot product\") entre deux vecteurs [1, 2, 3] et [4, 5, 6], ca donne 1*4 + 2*5 + 3*6 = 32. C'est une operation qu'on utilise partout dans le ML : c'est comme ca qu'un neurone calcule sa sortie.", { indent: true }));

children.push(p("On a aussi implemente le produit de Hadamard, qui est juste la multiplication element par element : [1, 2, 3] * [4, 5, 6] = [4, 10, 18]. Ca sert dans la backpropagation (on verra plus tard). Et l'argmax, qui renvoie l'index du plus grand element : dans [0.1, 0.7, 0.2], l'argmax c'est 1 (la deuxieme position). Ca sert a savoir quelle classe le modele a predite.", { indent: true }));

children.push(heading("5.2. Le type Matrix", HeadingLevel.HEADING_2));

children.push(p("La matrice, c'est un tableau a deux dimensions. On la stocke dans un seul tableau a plat (un Vec<f64>) avec la convention \"row-major\" : les elements de la premiere ligne d'abord, puis ceux de la deuxieme, etc. Pour acceder a l'element en ligne i, colonne j, on fait data[i * nombre_de_colonnes + j].", { indent: true }));

children.push(p("La multiplication de matrices, c'est l'operation la plus importante. On a implemente l'algorithme classique en trois boucles imbriquees. C'est pas le plus rapide qui existe (il y a des algorithmes optimises comme Strassen), mais pour les petites tailles qu'on utilise, ca suffit largement et c'est plus simple a comprendre.", { indent: true }));

children.push(p("Un truc qu'on a appris en codant ca : les erreurs de dimensions. Multiplier une matrice 2x3 par une matrice 5x2, ca n'a pas de sens mathematiquement. On a ajoute des verifications partout (des assert!) qui font planter le programme immediatement avec un message clair si les dimensions ne correspondent pas. Ca nous a fait gagner beaucoup de temps de debug.", { indent: true }));

// ═══ 6. ACTIVATIONS ═══
children.push(heading("6. Comprendre les fonctions d'activation", HeadingLevel.HEADING_1, true));

children.push(p("Quand on a commence a lire sur les reseaux de neurones, on est tombes sur les fonctions d'activation. Au debut, on comprenait pas bien a quoi elles servaient. Et puis on a eu le declic : sans fonction d'activation, un reseau de neurones, meme avec 100 couches, ne pourrait apprendre que des relations lineaires (des droites, des plans). C'est la fonction d'activation qui rend le reseau capable d'apprendre des choses plus complexes.", { indent: true }));

children.push(p("Pour expliquer simplement : chaque neurone calcule une somme ponderee de ses entrees (un produit scalaire), puis passe le resultat dans une fonction d'activation. C'est cette fonction qui decide \"est-ce que ce neurone s'active ou non\".", { indent: true }));

children.push(heading("6.1. Sigmoid : transformer un nombre en probabilite", HeadingLevel.HEADING_2));

children.push(p("La sigmoide, c'est la fonction sigma(x) = 1 / (1 + e^(-x)). Quel que soit le nombre qu'on lui donne (meme -1000 ou +1000), elle renvoie toujours un resultat entre 0 et 1. C'est pratique pour interpreter la sortie comme une probabilite.", { indent: true }));

children.push(p("Quand x vaut 0, sigma(0) = 0.5 (pile au milieu). Quand x est tres positif, sigma tend vers 1. Tres negatif, vers 0. C'est une transition douce, pas un seuil brutal.", { indent: true }));

children.push(p("Pour la backpropagation (qu'on verra apres), on a besoin de la derivee de chaque activation. La derivee de la sigmoide a une propriete super pratique : sigma'(x) = sigma(x) * (1 - sigma(x)). Du coup, une fois qu'on a calcule sigma(x), on peut calculer sa derivee quasi gratuitement.", { indent: true }));

children.push(heading("6.2. Tanh : comme sigmoid, mais centree sur zero", HeadingLevel.HEADING_2));

children.push(p("La tangente hyperbolique (tanh) est similaire a la sigmoide mais elle sort des valeurs entre -1 et +1 au lieu de 0 et 1. L'avantage, c'est que quand un neurone \"ne s'active pas\", il renvoie une valeur negative au lieu de juste zero. Ca aide l'entrainement parce que les gradients (les directions de mise a jour) sont mieux repartis autour de zero.", { indent: true }));

children.push(heading("6.3. ReLU : simple mais efficace (en theorie)", HeadingLevel.HEADING_2));

children.push(p("ReLU c'est la fonction la plus simple : si x est positif, on garde x ; si x est negatif, on met 0. C'est tout. C'est ce qu'on trouve dans la plupart des reseaux modernes parce que c'est rapide a calculer et que ca marche bien sur les grands reseaux.", { indent: true }));

children.push(p("Mais on a decouvert un probleme a nos depens : les \"neurones morts\". Si un neurone recoit toujours des valeurs negatives (a cause de poids mal initialises), sa sortie est toujours 0, son gradient est 0, et il ne peut plus jamais apprendre. C'est comme si il etait mort. Sur nos petits reseaux pour le XOR, ca arrivait assez souvent. Du coup, pour les petits problemes, on utilise plutot Sigmoid.", { indent: true }));

children.push(heading("6.4. Softmax : choisir une classe parmi plusieurs", HeadingLevel.HEADING_2));

children.push(p("Softmax, c'est la fonction qu'on met a la fin du reseau quand on fait de la classification. Elle prend un vecteur de scores (genre [2.0, 1.0, 0.5]) et les transforme en probabilites qui somment a 1 (genre [0.59, 0.24, 0.17]). La classe avec la plus grande probabilite, c'est la prediction du modele.", { indent: true }));

children.push(p("Un detail technique qu'on a appris en la codant : il faut soustraire le maximum avant de calculer les exponentielles. Sinon, si les scores sont grands (genre 500), exp(500) donne l'infini et le programme plante. C'est un truc classique de stabilite numerique qu'on a trouve en cherchant pourquoi nos tests plantaient.", { indent: true }));

children.push(heading("6.5. L'enum Activation", HeadingLevel.HEADING_2));

children.push(p("En Rust, un \"enum\" c'est un type qui peut prendre plusieurs formes. On a cree un enum Activation avec quatre variantes (Sigmoid, Tanh, ReLU, Softmax). Ca nous permet de choisir quelle activation utiliser sans dupliquer le code du MLP. On ecrit juste MLP::new(&[2, 8, 2]).with_activation(Activation::Sigmoid) et le reseau utilise Sigmoid pour ses couches cachees.", { indent: true }));

// ═══ 7. MODELE LINEAIRE ═══
children.push(heading("7. Le modele lineaire : notre premier pas en ML", HeadingLevel.HEADING_1, true));

children.push(p("Le modele lineaire, c'est la premiere chose qu'on a implementee. C'est le modele le plus simple possible : il cherche une relation de la forme y = w1*x1 + w2*x2 + ... + b. Par exemple, si on a des donnees ou y = 3x + 1, le modele va apprendre que w = 3 et b = 1.", { indent: true }));

children.push(heading("7.1. Comment il apprend : la descente de gradient", HeadingLevel.HEADING_2));

children.push(p("L'idee de base de l'apprentissage, c'est : le modele fait une prediction, on mesure a quel point il s'est trompe (l'erreur), et on ajuste les poids un tout petit peu dans la bonne direction pour que l'erreur diminue. On repete ca des milliers de fois, et normalement le modele s'ameliore progressivement.", { indent: true }));

children.push(p("Pour mesurer l'erreur, on utilise la MSE (Mean Squared Error) : on prend la difference entre la prediction et la vraie valeur, on l'eleve au carre, et on fait la moyenne. Elever au carre c'est important : ca penalise plus les grosses erreurs que les petites.", { indent: true }));

children.push(p("Pour trouver dans quelle direction ajuster les poids, on calcule le gradient : c'est la derivee de l'erreur par rapport a chaque poids. Le gradient nous dit \"si j'augmente ce poids, est-ce que l'erreur augmente ou diminue ?\". Ensuite on ajuste le poids dans la direction opposee au gradient (pour diminuer l'erreur). C'est pour ca qu'on appelle ca la \"descente de gradient\".", { indent: true }));

children.push(p("Le learning rate (taux d'apprentissage), c'est la taille du pas qu'on fait a chaque iteration. Trop grand, le modele oscille et ne converge pas. Trop petit, il met une eternite a apprendre. Trouver le bon learning rate, c'est un peu du tatonnement.", { indent: true }));

children.push(heading("7.2. Limites du lineaire", HeadingLevel.HEADING_2));

children.push(p("Un modele lineaire ne peut tracer que des droites (ou des hyperplans en dimension superieure). Si les donnees suivent une relation courbe ou si deux classes ne sont pas separables par une droite, il est impuissant. L'exemple classique c'est le XOR : les points (0,0) et (1,1) sont dans une classe, et (0,1) et (1,0) dans l'autre. Impossible de les separer avec une seule droite. C'est exactement pour ca qu'on a besoin du MLP.", { indent: true }));

// ═══ 8. LE MLP ═══
children.push(heading("8. Le MLP : la ou ca devient serieux", HeadingLevel.HEADING_1, true));

children.push(p("Le MLP (Multi-Layer Perceptron) c'est un reseau de neurones avec plusieurs couches. C'est ce qui nous permet de resoudre des problemes que le modele lineaire ne peut pas resoudre, comme le XOR. Coder le MLP from scratch, c'est probablement la partie ou on a le plus appris dans ce projet.", { indent: true }));

children.push(heading("8.1. C'est quoi concretement un MLP", HeadingLevel.HEADING_2));

children.push(p("On definit un MLP par la taille de ses couches. Par exemple [2, 8, 2] veut dire : 2 entrees, une couche cachee de 8 neurones, et 2 sorties. Chaque neurone d'une couche est connecte a tous les neurones de la couche suivante. Chaque connexion a un poids (un nombre), et chaque neurone a un biais.", { indent: true }));

children.push(p("Le \"2\" a la fin, c'est le nombre de classes. Si on veut classifier des images en chat/chien, on a 2 sorties : une pour chat, une pour chien. Le softmax a la fin transforme ces deux scores en probabilites (par exemple : 80% chat, 20% chien).", { indent: true }));

children.push(heading("8.2. Le forward pass : de l'entree a la prediction", HeadingLevel.HEADING_2));

children.push(p("Le forward pass, c'est le trajet de l'entree a travers toutes les couches. A chaque couche, chaque neurone calcule le produit scalaire de ses poids avec les sorties de la couche precedente, ajoute son biais, et passe le tout dans la fonction d'activation. La derniere couche utilise softmax pour donner des probabilites.", { indent: true }));

children.push(p("En code, c'est une simple boucle sur les couches. Ce qui nous a pris du temps, c'est de bien gerer les dimensions : si la couche a 8 neurones en entree et 4 en sortie, il faut 8*4 = 32 poids plus 4 biais. Une erreur d'indice et plus rien ne marche.", { indent: true }));

children.push(heading("8.3. L'initialisation des poids", HeadingLevel.HEADING_2));

children.push(p("Un truc qu'on ne soupconnait pas : l'initialisation des poids est critique. Si on initialise tous les poids a zero, tous les neurones calculent la meme chose et le reseau n'apprend rien (on dit qu'il \"casse la symetrie\"). On les initialise aleatoirement dans [-0.5, 0.5]. Si les poids sont trop grands, les calculs explosent. Trop petits, les gradients disparaissent.", { indent: true }));

// ═══ 9. BACKPROP ═══
children.push(heading("9. La backpropagation : ce qu'on a appris en la codant", HeadingLevel.HEADING_1, true));

children.push(p("La backpropagation, c'est l'algorithme qui permet d'entrainer un reseau de neurones. C'est probablement le concept le plus important qu'on a appris dans ce projet, et aussi le plus dur a implementer correctement.", { indent: true }));

children.push(heading("9.1. L'idee en simple", HeadingLevel.HEADING_2));

children.push(p("Quand le reseau fait une prediction, il se trompe d'un certain montant (l'erreur). La question c'est : comment ajuster chacun des poids du reseau pour reduire cette erreur ? Le probleme, c'est que dans un reseau a plusieurs couches, les poids de la premiere couche ont un impact indirect sur la sortie, a travers toutes les couches suivantes.", { indent: true }));

children.push(p("La backpropagation resout ca en utilisant la regle de la chaine (qu'on avait vue en cours de maths mais sans vraiment comprendre a quoi ca servait). En gros, on calcule l'erreur a la sortie, puis on la propage \"en arriere\" couche par couche pour savoir quel poids est responsable de quelle partie de l'erreur.", { indent: true }));

children.push(heading("9.2. Comment on l'a implementee", HeadingLevel.HEADING_2));

children.push(p("L'algorithme se deroule en plusieurs etapes. D'abord, on fait un forward pass en gardant en memoire toutes les valeurs intermediaires (les valeurs avant et apres chaque activation). Ensuite, a la derniere couche, on calcule l'ecart entre la prediction et la cible : c'est le \"delta\". Puis on remonte couche par couche : pour chaque couche, on utilise le delta pour mettre a jour les poids et biais, et on calcule le delta de la couche precedente.", { indent: true }));

children.push(p("Ce qui est elegant, c'est que pour la combinaison softmax + erreur de classification, le delta initial est simplement sortie - cible. C'est une formule tres simple pour un calcul qui pourrait etre tres complique. On ne l'avait pas compris au debut et on avait essaye de calculer la derivee du softmax directement, ce qui donnait des formules beaucoup plus longues.", { indent: true }));

children.push(heading("9.3. Nos galeres avec la backprop", HeadingLevel.HEADING_2));

children.push(p("On va etre honnetes : ca nous a pris plusieurs jours pour que ca marche. Le premier probleme, c'est que quand la backprop est buguee, le reseau n'apprend tout simplement pas. Mais il ne plante pas non plus : il donne juste des predictions aleatoires. Du coup, c'est dur de savoir ou est le bug.", { indent: true }));

children.push(p("Notre methode pour debugger : on a teste sur un reseau a une seule couche (qui devrait se comporter comme un modele lineaire), verifie que les gradients etaient corrects a la main sur un exemple simple, puis ajoute les couches une par une. C'est comme ca qu'on a trouve qu'on avait inverse un indice dans la propagation du delta.", { indent: true }));

// ═══ 10. SERIALISATION ═══
children.push(heading("10. Sauvegarder un modele : la serialisation", HeadingLevel.HEADING_1, true));

children.push(p("La serialisation, c'est le fait de transformer un objet en memoire (notre modele MLP avec tous ses poids) en un fichier qu'on peut sauvegarder sur le disque et recharger plus tard. Sans ca, il faudrait re-entrainer le modele a chaque fois.", { indent: true }));

children.push(p("On a utilise serde, qui est LA bibliotheque de serialisation en Rust. Le principe est simple : on ajoute un attribut #[derive(Serialize, Deserialize)] sur nos structs, et serde genere automatiquement le code pour les convertir en JSON ou en binaire.", { indent: true }));

children.push(heading("10.1. Format JSON", HeadingLevel.HEADING_2));

children.push(p("Le JSON c'est du texte lisible par un humain. Un modele sauvegarde en JSON ressemble a un fichier de configuration : on voit les poids, les biais, les tailles des couches. C'est pratique pour verifier que le modele est correct, mais c'est verbeux (un nombre comme 0.123456789 prend 11 caracteres au lieu de 8 octets).", { indent: true }));

children.push(heading("10.2. Format binaire", HeadingLevel.HEADING_2));

children.push(p("Pour des modeles plus gros, on a ajoute bincode qui sauvegarde en binaire. C'est beaucoup plus compact et rapide, mais on ne peut pas lire le fichier avec un editeur de texte. En pratique, on utilise le JSON pour le debug et le binaire pour la production.", { indent: true }));

children.push(p("On a ecrit des tests qui verifient que sauvegarder puis recharger un modele donne exactement les memes predictions. C'est important parce qu'avec les nombres a virgule, on peut avoir des pertes de precision.", { indent: true }));

// ═══ 11. API REST ═══
children.push(heading("11. L'API REST : rendre le modele accessible", HeadingLevel.HEADING_1, true));

children.push(p("Une API REST, c'est un serveur web qui repond a des requetes HTTP. Dans notre cas, on peut lui envoyer une image et il renvoie la prediction du modele. C'est comme ca que les services de ML fonctionnent dans la vraie vie : le modele tourne sur un serveur, et les clients (une app mobile, un site web) envoient des requetes.", { indent: true }));

children.push(p("On a utilise Axum, un framework web en Rust qui fonctionne de maniere asynchrone (il peut gerer plusieurs requetes en meme temps sans bloquer). C'est Nina qui a fait cette partie.", { indent: true }));

children.push(heading("11.1. Les endpoints", HeadingLevel.HEADING_2));

const apiTable = new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [1200, 2000, 6160],
    rows: [
        new TableRow({ children: [
            cell("Methode", { width: 1200, header: true }),
            cell("Route", { width: 2000, header: true }),
            cell("Ce que ca fait", { width: 6160, header: true }),
        ]}),
        new TableRow({ children: [ cell("POST", { width: 1200 }), cell("/predict", { width: 2000 }), cell("On envoie une image en base64, on recoit la classe predite et le score de confiance", { width: 6160 }) ]}),
        new TableRow({ children: [ cell("POST", { width: 1200 }), cell("/train", { width: 2000 }), cell("On lance l'entrainement d'un nouveau modele avec les parametres donnes", { width: 6160 }) ]}),
        new TableRow({ children: [ cell("GET", { width: 1200 }), cell("/models", { width: 2000 }), cell("On recupere la liste de tous les modeles disponibles", { width: 6160 }) ]}),
    ],
});
children.push(apiTable);
children.push(p(""));

children.push(p("Les modeles sont stockes sur le disque en JSON et charges en memoire au demarrage du serveur. On utilise un systeme de verrou (RwLock) qui permet a plusieurs requetes de lire les modeles en meme temps, mais qui bloque tout le monde quand on modifie un modele. C'est un pattern classique en programmation concurrente.", { indent: true }));

// ═══ 12. BINDINGS PYTHON ═══
children.push(heading("12. Les bindings Python : connecter Rust et Python", HeadingLevel.HEADING_1, true));

children.push(p("Les bindings, c'est un pont entre deux langages. Concretement, ca permet d'ecrire \"import visionai\" en Python et d'utiliser nos modeles Rust comme si c'etaient des objets Python normaux. C'est comme ca que fonctionnent NumPy (ecrit en C) ou PyTorch (ecrit en C++).", { indent: true }));

children.push(p("On utilise PyO3, une bibliotheque qui genere automatiquement le code de liaison entre Rust et Python. On annote nos fonctions Rust avec des attributs speciaux (#[pyclass], #[pymethods]) et PyO3 genere le code necessaire pour que Python puisse les appeler.", { indent: true }));

children.push(p("Un exemple concret d'utilisation :", { indent: true }));
children.push(p("import visionai", { size: 22, after: 40 }));
children.push(p("model = visionai.create_model('mlp', {'layer_sizes': [2, 8, 2]})", { size: 22, after: 40 }));
children.push(p("visionai.train(model, inputs, targets, epochs=1000, lr=0.5)", { size: 22, after: 40 }));
children.push(p("result = visionai.predict(model, [1.0, 0.0])", { size: 22, after: 200 }));

children.push(p("C'est Nina qui a fait cette partie. Le plus complique, c'etait de gerer la conversion des types : une liste Python de flottants doit devenir un Vec<f64> en Rust, et inversement pour le resultat.", { indent: true }));

// ═══ 13. CLIENT WEB ═══
children.push(heading("13. L'application cliente web", HeadingLevel.HEADING_1, true));

children.push(p("L'application cliente, c'est une page web toute simple qui permet de tester le modele visuellement. On uploade une image, on clique sur un bouton, et on voit la prediction. C'est fait en HTML/CSS/JavaScript pur, sans framework. Pour un truc aussi simple, React ou Vue ca aurait ete overkill.", { indent: true }));

children.push(p("L'image est convertie en base64 (un format texte qui represente des donnees binaires) directement dans le navigateur, puis envoyee au serveur API. Le resultat s'affiche avec la classe predite et une barre de confiance. On a aussi un mode drag & drop pour uploader les images plus facilement.", { indent: true }));

// ═══ 14. TESTS ═══
children.push(heading("14. Tests : comment on verifie que ca marche", HeadingLevel.HEADING_1, true));

children.push(p("On a 40 tests au total. C'est pas enorme pour un vrai framework, mais pour un projet d'apprentissage, on trouve que c'est correct. Ca couvre les maths de base, les modeles, et les cas classiques du ML.", { indent: true }));

children.push(heading("14.1. Les tests logiques : XOR, AND, OR", HeadingLevel.HEADING_2));

children.push(p("Ce sont les problemes de reference pour tester un reseau de neurones. XOR (le \"ou exclusif\") est le plus interessant parce qu'il n'est pas lineairement separable : on ne peut pas tracer une droite qui separe les cas \"vrai\" des cas \"faux\". Si notre MLP arrive a apprendre le XOR, c'est que la backpropagation fonctionne.", { indent: true }));

const testTable = new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [1600, 2600, 2000, 3160],
    rows: [
        new TableRow({ children: [
            cell("Probleme", { width: 1600, header: true }),
            cell("Architecture", { width: 2600, header: true }),
            cell("Activation", { width: 2000, header: true }),
            cell("Resultat", { width: 3160, header: true }),
        ]}),
        new TableRow({ children: [ cell("XOR", { width: 1600 }), cell("[2, 16, 2]", { width: 2600 }), cell("Sigmoid", { width: 2000 }), cell("Converge (5 tentatives max)", { width: 3160 }) ]}),
        new TableRow({ children: [ cell("AND", { width: 1600 }), cell("[2, 8, 2]", { width: 2600 }), cell("Sigmoid", { width: 2000 }), cell("Converge", { width: 3160 }) ]}),
        new TableRow({ children: [ cell("OR", { width: 1600 }), cell("[2, 8, 2]", { width: 2600 }), cell("Sigmoid", { width: 2000 }), cell("Converge", { width: 3160 }) ]}),
        new TableRow({ children: [ cell("Multi-classes", { width: 1600 }), cell("[2, 16, 3]", { width: 2600 }), cell("Sigmoid", { width: 2000 }), cell("3 clusters OK", { width: 3160 }) ]}),
        new TableRow({ children: [ cell("Sinus", { width: 1600 }), cell("[1, 16, 2]", { width: 2600 }), cell("Sigmoid", { width: 2000 }), cell("Distingue sin>0 vs sin<0", { width: 3160 }) ]}),
    ],
});
children.push(testTable);
children.push(p(""));

children.push(heading("14.2. Le probleme des tests non-deterministes", HeadingLevel.HEADING_2));

children.push(p("On a decouvert que les tests de ML sont compliques a faire. Comme les poids sont initialises aleatoirement, parfois le reseau converge, parfois non, meme avec exactement le meme code. Un test qui passe 9 fois sur 10 et echoue la 10eme, c'est tres frustrant.", { indent: true }));

children.push(p("Notre solution : on autorise jusqu'a 5 tentatives par test. Si apres 5 essais le reseau n'a toujours pas converge, la c'est probablement un vrai bug. En pratique, la plupart des tests passent du premier coup.", { indent: true }));

// ═══ 15. RESULTATS ═══
children.push(heading("15. Nos resultats (et ce qu'on en comprend)", HeadingLevel.HEADING_1, true));

children.push(p("Nos 40 tests passent. C'est notre principale metrique de succes pour ce livrable. Ca prouve que nos implementations mathematiques sont correctes et que le MLP est capable d'apprendre.", { indent: true }));

children.push(heading("15.1. Ce qui marche bien", HeadingLevel.HEADING_2));

children.push(p("Le XOR converge en quelques milliers d'iterations. C'est le resultat dont on est le plus contents parce que ca valide toute la chaine : forward pass, softmax, backpropagation, mise a jour des poids. Si un seul de ces elements etait buguee, le XOR ne convergerait pas.", { indent: true }));

children.push(p("La serialisation marche parfaitement : on sauvegarde un modele, on le recharge, et les predictions sont identiques a 1e-10 pres.", { indent: true }));

children.push(heading("15.2. Ce qu'on n'a pas encore fait", HeadingLevel.HEADING_2));

children.push(p("On n'a pas encore teste sur de vraies images. Nos tests utilisent des donnees synthetiques (des vecteurs de 2-3 dimensions). Tester sur un vrai dataset comme MNIST (des images de chiffres manuscrits) c'est prevu pour la semaine prochaine. Le script de preprocessing est pret, il faut juste telecharger les donnees et lancer les experiments.", { indent: true }));

children.push(p("On n'a pas non plus fait d'analyse des phenomenes d'apprentissage (sur-apprentissage, sous-apprentissage, impact des hyperparametres). C'est la partie experimentation qui est assignee a Ali.", { indent: true }));

// ═══ 16. DIFFICULTES ═══
children.push(heading("16. Ce qui nous a pose probleme", HeadingLevel.HEADING_1, true));

children.push(p("Ce serait mentir de dire que tout s'est bien passe. On a galere sur plusieurs points, et on pense que c'est important d'en parler parce que c'est aussi la qu'on a appris.", { indent: true }));

children.push(heading("16.1. Apprendre Rust et le ML en meme temps", HeadingLevel.HEADING_2));

children.push(p("C'etait probablement notre plus grosse difficulte. Quand le code ne marche pas, est-ce que c'est parce qu'on a mal compris l'algorithme, ou parce qu'on ne sait pas ecrire du Rust ? Souvent c'etait un peu des deux. On a du alterner entre lire des cours de ML et lire la doc Rust, ce qui ralentissait les choses.", { indent: true }));

children.push(heading("16.2. La backpropagation", HeadingLevel.HEADING_2));

children.push(p("Les equations de la backpropagation sur le papier, ca va. Mais les traduire en code sans se tromper d'indice, de signe ou de transposition, c'est une autre histoire. On a passe pas mal de temps a debugger avec des reseaux tout petits (2 neurones) pour verifier les calculs a la main.", { indent: true }));

children.push(heading("16.3. Le borrow checker", HeadingLevel.HEADING_2));

children.push(p("Le borrow checker de Rust (le systeme qui verifie la memoire) nous a bloque sur la backpropagation. Le probleme : dans notre boucle, on voulait lire les poids de la couche l+1 tout en modifiant les poids de la couche l. Rust interdisait ca parce que les deux couches sont dans le meme tableau. On a du restructurer le code pour contourner le probleme.", { indent: true }));

children.push(heading("16.4. Trouver les bons hyperparametres", HeadingLevel.HEADING_2));

children.push(p("Le learning rate, le nombre de couches, le nombre de neurones par couche, le nombre d'epochs... ca fait beaucoup de parametres a regler, et on n'avait aucune intuition au debut. On a procede par essai-erreur, ce qui est long mais formateur. Par exemple, on a compris qu'un learning rate de 0.001 c'est trop petit pour 4 exemples d'entrainement (le XOR), et qu'il faut monter a 0.5 voire 1.0.", { indent: true }));

// ═══ 17. ETAT D'AVANCEMENT ═══
children.push(heading("17. Etat d'avancement et prochaines etapes", HeadingLevel.HEADING_1, true));

children.push(p("Ce rapport correspond a un livrable intermediaire. Le code est fonctionnel et teste, mais il reste du travail pour la soutenance finale.", { indent: true }));

children.push(heading("17.1. Ce qu'on a fait", HeadingLevel.HEADING_2));

children.push(p("Cote code, les fondations sont solides : le core ML (maths, modeles, activations, serialisation), l'API, les bindings Python, le client web, et 40 tests. C'est la partie la plus technique et elle est terminee.", { indent: true }));

children.push(heading("17.2. Ce qu'il reste a faire", HeadingLevel.HEADING_2));

children.push(p([{ text: "Les notebooks d'experimentation. ", bold: true }, { text: "Ali doit finir les notebooks Jupyter qui testent les modeles sur des donnees reelles et produisent les courbes (loss vs epochs, matrices de confusion, decision boundaries). Le code est pret cote Rust, il faut juste l'exploiter cote Python." }]));

children.push(p([{ text: "Le dataset d'images. ", bold: true }, { text: "Le script de preprocessing est ecrit, mais on n'a pas encore telecharge et prepare un vrai dataset. C'est planifie pour cette semaine." }]));

children.push(p([{ text: "L'analyse des phenomenes d'apprentissage. ", bold: true }, { text: "On doit montrer ce qui se passe quand le modele est trop simple (sous-apprentissage) ou trop complexe (sur-apprentissage), et l'impact des hyperparametres. Ca demande de faire tourner plein d'entrainements avec des parametres differents et de comparer les courbes." }]));

children.push(p([{ text: "La preparation de la soutenance. ", bold: true }, { text: "Preparer les slides, choisir les demos a montrer, s'entrainer a presenter." }]));

children.push(heading("17.3. Ce qu'on retient", HeadingLevel.HEADING_2));

children.push(p("Meme si le projet n'est pas fini, on a deja beaucoup appris. On sait maintenant ce que fait un reseau de neurones sous le capot, comment fonctionne la descente de gradient, pourquoi l'initialisation des poids est importante, et ce qu'est un gradient qui \"disparait\". Ce sont des choses qu'on n'aurait jamais comprises en utilisant juste scikit-learn ou PyTorch.", { indent: true }));

children.push(p("On a aussi appris Rust, ce qui n'etait pas prevu au depart comme objectif principal mais qui s'est avere etre une grosse partie du travail (et de l'apprentissage).", { indent: true }));

// ═══ ANNEXES ═══
children.push(heading("Annexes", HeadingLevel.HEADING_1, true));

children.push(heading("A. Repartition du travail", HeadingLevel.HEADING_2));

const teamTable = new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2000, 7360],
    rows: [
        new TableRow({ children: [
            cell("Membre", { width: 2000, header: true }),
            cell("Responsabilites", { width: 7360, header: true }),
        ]}),
        new TableRow({ children: [
            cell("Valentin", { width: 2000 }),
            cell("Core ML : maths (Vector, Matrix), fonctions d'activation, modele MLP (forward, backprop), serialisation serde, tests unitaires et d'integration (40 tests), application cliente web, rapport", { width: 7360 }),
        ]}),
        new TableRow({ children: [
            cell("Ali", { width: 2000 }),
            cell("Dataset : preprocessing des images, notebooks d'experimentation, analyse des phenomenes d'apprentissage (a finir)", { width: 7360 }),
        ]}),
        new TableRow({ children: [
            cell("Nina", { width: 2000 }),
            cell("API server REST (Axum), bindings Python (PyO3), fonctions utilitaires (create_model, train, predict, save, load)", { width: 7360 }),
        ]}),
    ],
});
children.push(teamTable);
children.push(p(""));

children.push(heading("B. Arborescence du projet", HeadingLevel.HEADING_2));

const treeLines = [
    "VisionAI/",
    "+-- Cargo.toml              (workspace)",
    "+-- core_lib/",
    "|   +-- src/",
    "|   |   +-- math/",
    "|   |   |   +-- vector.rs      (173 lignes)",
    "|   |   |   +-- matrix.rs      (125 lignes)",
    "|   |   |   +-- activations.rs (160 lignes)",
    "|   |   +-- models/",
    "|   |   |   +-- linear.rs      (108 lignes)",
    "|   |   |   +-- mlp.rs         (176 lignes)",
    "|   |   +-- optim/",
    "|   |       +-- gradient_descent.rs",
    "|   +-- tests/",
    "|       +-- integration_tests.rs (40 tests)",
    "+-- api_server/",
    "|   +-- src/main.rs            (350+ lignes)",
    "+-- python_binding/",
    "|   +-- src/lib.rs             (303 lignes)",
    "+-- client/",
    "|   +-- index.html             (app web)",
    "+-- notebooks/",
    "    +-- preprocess_dataset.py",
    "    +-- analyse_dataset.ipynb",
    "    +-- test_linear.ipynb",
];
treeLines.forEach(line => {
    children.push(new Paragraph({
        children: [new TextRun({ text: line, size: 20, font: "Consolas", color: "444444" })],
        spacing: { after: 30 },
    }));
});

children.push(heading("C. Glossaire", HeadingLevel.HEADING_2));

children.push(p("Voici les termes techniques qu'on a decouverts pendant le projet :", { indent: true }));

const glossTable = new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2800, 6560],
    rows: [
        new TableRow({ children: [
            cell("Terme", { width: 2800, header: true }),
            cell("Ce que ca veut dire", { width: 6560, header: true }),
        ]}),
        new TableRow({ children: [ cell("Backpropagation", { width: 2800 }), cell("Algorithme pour calculer comment ajuster les poids d'un reseau de neurones", { width: 6560 }) ]}),
        new TableRow({ children: [ cell("Borrow checker", { width: 2800 }), cell("Systeme de Rust qui verifie a la compilation qu'il n'y a pas d'erreurs de memoire", { width: 6560 }) ]}),
        new TableRow({ children: [ cell("Crate", { width: 2800 }), cell("L'equivalent d'un package/bibliotheque en Rust", { width: 6560 }) ]}),
        new TableRow({ children: [ cell("Epoch", { width: 2800 }), cell("Un passage complet sur l'ensemble des donnees d'entrainement", { width: 6560 }) ]}),
        new TableRow({ children: [ cell("Forward pass", { width: 2800 }), cell("Le trajet des donnees de l'entree a la sortie du reseau", { width: 6560 }) ]}),
        new TableRow({ children: [ cell("Gradient", { width: 2800 }), cell("Direction dans laquelle modifier les poids pour reduire l'erreur", { width: 6560 }) ]}),
        new TableRow({ children: [ cell("Learning rate", { width: 2800 }), cell("Taille du pas de mise a jour des poids a chaque iteration", { width: 6560 }) ]}),
        new TableRow({ children: [ cell("MLP", { width: 2800 }), cell("Multi-Layer Perceptron, un reseau de neurones a plusieurs couches", { width: 6560 }) ]}),
        new TableRow({ children: [ cell("MSE", { width: 2800 }), cell("Mean Squared Error, une mesure de l'erreur de prediction", { width: 6560 }) ]}),
        new TableRow({ children: [ cell("Neurone mort", { width: 2800 }), cell("Un neurone bloque a zero qui ne peut plus apprendre (probleme de ReLU)", { width: 6560 }) ]}),
        new TableRow({ children: [ cell("One-hot encoding", { width: 2800 }), cell("Representer une classe par un vecteur avec un seul 1 (ex: classe 2 sur 3 = [0,1,0])", { width: 6560 }) ]}),
        new TableRow({ children: [ cell("Serialisation", { width: 2800 }), cell("Transformer un objet en memoire en fichier sur le disque", { width: 6560 }) ]}),
        new TableRow({ children: [ cell("SGD", { width: 2800 }), cell("Stochastic Gradient Descent, la methode d'optimisation qu'on utilise", { width: 6560 }) ]}),
        new TableRow({ children: [ cell("Softmax", { width: 2800 }), cell("Fonction qui transforme des scores en probabilites (somme = 1)", { width: 6560 }) ]}),
    ],
});
children.push(glossTable);
children.push(p(""));

children.push(heading("D. References", HeadingLevel.HEADING_2));

children.push(p("Ce qu'on a utilise pour apprendre :", { indent: true }));
children.push(p([{ text: "The Rust Programming Language ", italics: true }, { text: "(le \"Rust Book\") - La doc officielle de Rust, on l'a consultee quasi tous les jours." }]));
children.push(p([{ text: "Neural Networks and Deep Learning ", italics: true }, { text: "de Michael Nielsen - Un site web gratuit qui explique la backpropagation de maniere tres visuelle." }]));
children.push(p([{ text: "3Blue1Brown - serie Neural Networks ", italics: true }, { text: "sur YouTube - Des videos avec des animations qui rendent la backprop intuitive." }]));
children.push(p([{ text: "CS231n (Stanford) ", italics: true }, { text: "- Les notes de cours en ligne pour les derivees des fonctions d'activation." }]));
children.push(p([{ text: "Documentation PyO3 et Axum ", italics: true }, { text: "- Les docs officielles pour les bindings Python et le serveur web." }]));

children.push(new Paragraph({ spacing: { before: 600 } }));
children.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "--- Fin du livrable intermediaire ---", size: 24, font: "Calibri", color: "888888", italics: true })],
}));

// ═══ BUILD ═══
const doc = new Document({
    styles: {
        default: { document: { run: { font: "Calibri", size: 24 } } },
        paragraphStyles: [
            { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 36, bold: true, font: "Calibri", color: "1F3864" },
              paragraph: { spacing: { before: 400, after: 200 }, outlineLevel: 0 } },
            { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 30, bold: true, font: "Calibri", color: "2E75B6" },
              paragraph: { spacing: { before: 300, after: 200 }, outlineLevel: 1 } },
        ],
    },
    sections: [{
        properties: {
            page: {
                size: { width: 11906, height: 16838 },
                margin: { top: 1440, right: 1260, bottom: 1440, left: 1260 },
            },
        },
        headers: {
            default: new Header({
                children: [new Paragraph({
                    alignment: AlignmentType.RIGHT,
                    children: [new TextRun({ text: "VisionAI - Livrable intermediaire", size: 18, font: "Calibri", color: "999999", italics: true })],
                    border: { bottom: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC", space: 4 } },
                })],
            }),
        },
        footers: {
            default: new Footer({
                children: [new Paragraph({
                    alignment: AlignmentType.CENTER,
                    children: [
                        new TextRun({ text: "Page ", size: 18, font: "Calibri", color: "999999" }),
                        new TextRun({ children: [PageNumber.CURRENT], size: 18, font: "Calibri", color: "999999" }),
                    ],
                })],
            }),
        },
        children,
    }],
});

Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync("client/rapport_visionai.docx", buffer);
    console.log("Rapport genere: client/rapport_visionai.docx (" + (buffer.length / 1024).toFixed(1) + " KB)");
});
