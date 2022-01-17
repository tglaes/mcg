const fs = require('fs')

let outputFileName = "matrix.csv";

// The dimension of the matrix and the vector
const matrixDimension = 10;
// The minimal value of an element on the diagonal
const minValueDiagonalElement = 100;
// The maximal value of an element on the diagonal
const maxValueDiagonalElement = 200;
// The maximal value of an element which is not on the diagonal
const maxValueNonDiagonalElement = 10;
// The amount of element per row unequal to zero (How many numbers unequal to zero should the row have)
let NUMBER_OF_NON_ZERO_ENTRIES_PER_ROW = 3;
// The amount of elements unequal to zero is NUMBER_OF_NON_ZERO_ENTRIES_PER_ROW + 1 (beacuse of the element on the diagonal)

// COO rows
// The total number of COO rows in the matrix (set to zero to not use COO rows)
let NUMBER_OF_COO_ROWS = 5;
// The amount of numbers unequal to zero in a COO row
let NUMBER_OF_NON_ZERO_ENTRIES_IN_COO_ROW = 8;
// An array with the indicies of COO rows
let cooRowIndexArray;

// The total amount of zeroes in the matrix (DO NOT SET, IT IS CALCULATED)
let numberOfZeroes = 0;

const matrix = Array.from(Array(matrixDimension), () => new Array(matrixDimension));
const vector = Array(matrixDimension);

generateMatrix();
generateVector();
//printVectorAndMatrix();
//testMatrixAndVector();
//writeMatrixAndVectorToFile();
//writeMatrixAndVectorToFileCsr();
writeMatrixAndVectorToFileEllCoo();

function generateMatrix() {

    // Generate the indicies for the COO rows
    let cooRowIndicies = generateRandomIndicies(-1, NUMBER_OF_COO_ROWS);
    cooRowIndexArray = cooRowIndicies.slice();
    let currentCooRowIndex = cooRowIndicies.pop();

    let start = new Date().getTime() / 1000;
    for (let i = 0; i < matrixDimension; i++) {
        let indicies = generateRandomIndicies(i, NUMBER_OF_NON_ZERO_ENTRIES_PER_ROW);
        let currentIndex = indicies.pop();
        if (currentCooRowIndex === i) {
            indicies = generateRandomIndicies(i, NUMBER_OF_NON_ZERO_ENTRIES_IN_COO_ROW);
            currentIndex = indicies.pop();
            currentCooRowIndex = cooRowIndicies.pop();
        }

        for (let j = 0; j < matrixDimension; j++) {

            if (i === j) {
                matrix[i][j] = roundNumber(getRandomNumberInRange(minValueDiagonalElement, maxValueDiagonalElement));
            } else if (j === currentIndex) {
                matrix[i][j] = roundNumber(getRandomNumberInRange(0, maxValueNonDiagonalElement));
                currentIndex = indicies.pop();
            } else {
                matrix[i][j] = 0;
                numberOfZeroes++;
            }
        }
    }
    let end = new Date().getTime() / 1000;
    console.log("Done generating matrix in " + roundNumber(end - start) + " seconds with " + numberOfZeroes + " zero elements");
}

function generateVector() {
    for (let i = 0; i < matrixDimension; i++) {
        vector[i] = roundNumber(getRandomNumberInRange(minValueDiagonalElement, maxValueDiagonalElement));
    }
    console.log("Done generating vector")
}

function generateRandomIndicies(excluded, amount) {

    let indicies = [];

    for (let i = 0; i < amount; i++) {
        let index = Math.floor(getRandomNumberInRange(0, matrixDimension))
        if (index !== excluded && (indicies.findIndex(x => x === index) === -1)) {
            indicies.push(index);
        } else {
            i--;
        }
    }
    //console.log(indicies.sort((a, b) => a - b).reverse());
    return indicies.sort((a, b) => a - b).reverse();
}

function printVectorAndMatrix() {
    for (let i = 0; i < matrixDimension; i++) {
        for (let j = 0; j < matrixDimension; j++) {
            process.stdout.write(matrix[i][j] + " ");
        }
        console.log("");
    }

    for (let i = 0; i < matrixDimension; i++) {
        console.log(vector[i]);
    }
}

function testMatrixAndVector() {
    let start = new Date().getTime() / 1000;

    for (let i = 0; i < matrixDimension; i++) {
        for (let j = i + 1; j < matrixDimension; j++) {

            // This array denotes the result of matrix[i][0] / matrix [j][0], matrix[i][1] / matrix [j][1], etc.
            let rowCoefficientArray = Array(matrixDimension + 1);
            for (let l = 0; l < matrixDimension; l++) {
                rowCoefficientArray[l] = matrix[i][l] / matrix[j][l];
            }
            // Add the vector
            rowCoefficientArray[matrixDimension] = vector[i] / vector[j];

            // Check if all the values in the rowCoefficientArray are the same. If yes rang(matrix) < dimension.
            let matrixRowAreDependent = true;
            for (let k = 0; k < matrixDimension + 1; k++) {
                if (k === 0) {
                    continue;
                }
                if (rowCoefficientArray[0] !== rowCoefficientArray[k]) {
                    matrixRowAreDependent = false;
                    break;
                }
            }
            if (matrixRowAreDependent) {
                console.log("Matrix row " + i + " is a multiplicate of row " + j)
            }
        }
        if ((i % (matrixDimension / 10)) === 0) {
            console.log(i + " rows tested");
        }
    }
    console.log(matrixDimension + " rows tested (matrix has a single solution)");
    let end = new Date().getTime() / 1000;
    console.log("Done testing matrix in " + roundNumber(end - start) + " seconds");
}

function writeMatrixAndVectorToFile() {
    outputFileName = "matrix_" + matrixDimension + ".csv";
    let start = new Date().getTime() / 1000;
    fs.writeFileSync(outputFileName, "" + matrixDimension + "\n", function () { });
    for (let i = 0; i < matrixDimension; i++) {
        let row = "";
        for (let j = 0; j < matrixDimension; j++) {
            row += "," + matrix[i][j];
        }
        row = row.substring(1, row.length);
        row = row + "\n";

        fs.appendFileSync(outputFileName, row, function (err) { })
    }

    writeArrayToFile(vector, false);

    let end = new Date().getTime() / 1000;
    console.log("Done writing matrix and vector to file in " + roundNumber(end - start) + " seconds");
}

function writeMatrixAndVectorToFileCsr() {
    outputFileName = "matrix_csr_" + matrixDimension + ".csv";
    let v = [];
    let col_index = [];
    let row_index = [];
    row_index.push(0);

    let start = new Date().getTime() / 1000;

    for (let i = 0; i < matrixDimension; i++) {
        let numberOfNonZeroesInCurrentRow = 0;
        for (let j = 0; j < matrixDimension; j++) {
            if (matrix[i][j] !== 0) {
                v.push(matrix[i][j]);
                col_index.push(j);
                numberOfNonZeroesInCurrentRow++;
            }
        }
        row_index.push(row_index[row_index.length - 1] + numberOfNonZeroesInCurrentRow);
    }

    fs.writeFileSync(outputFileName, "" + matrixDimension + "," + v.length + "," + row_index.length + "\n", function () { });
    writeArrayToFile(v, true);
    writeArrayToFile(col_index, true);
    writeArrayToFile(row_index, true);
    writeArrayToFile(vector, false);

    let end = new Date().getTime() / 1000;
    console.log("Done writing matrix and vector to file in " + roundNumber(end - start) + " seconds");
}

function writeMatrixAndVectorToFileEllCoo() {
    outputFileName = "matrix_ell_coo_" + matrixDimension + ".csv";
    // ELL
    let dataEll = [];
    let colsEll = [];
    let currentColumnIndexMap = new Map();

    // COO
    let dataCOO = [];
    let rowsCOO = [];
    let colsCOO = [];

    let start = new Date().getTime() / 1000;

    // ELL
    for (let j = 0; j < matrixDimension; j++) {
        for (let i = 0; i < matrixDimension; i++) {
            if (cooRowIndexArray.findIndex(x => x === i) !== -1) {
                continue;
            }
            if (currentColumnIndexMap.get(i) >= j) {
                j = currentColumnIndexMap.get(i);
            }
            for (let k = j; k < matrixDimension; k++) {
                if (matrix[i][k] !== 0) {
                    dataEll.push(matrix[i][k]);
                    colsEll.push(k);
                    currentColumnIndexMap.set(i, k + 1);
                    break;
                }
            }
        }
    }

    // COO
    for (let i = 0; i < cooRowIndexArray.length; i++) {
        let rowIndex = cooRowIndexArray.pop();
        for (let j = 0; j < matrixDimension; j++) {
            if (matrix[rowIndex][j] !== 0) {
                dataCOO.push(matrix[rowIndex][j]);
                rowsCOO.push(rowIndex);
                colsCOO.push(j);
            }
        }
    }
    /*
        for (let i = 0; i < dataEll.length; i++) {
            process.stdout.write(dataEll[i] + ", ");
        }
        console.log("");
        for (let i = 0; i < colsEll.length; i++) {
            process.stdout.write(colsEll[i] + ", ");
        }
        console.log("");
    
        for (let i = 0; i < dataCOO.length; i++) {
            process.stdout.write(dataCOO[i] + ", ");
        }
        console.log("");
        for (let i = 0; i < rowsCOO.length; i++) {
            process.stdout.write(rowsCOO[i] + ", ");
        }
        console.log("");
    
        for (let i = 0; i < colsCOO.length; i++) {
            process.stdout.write(colsCOO[i] + ", ");
        }
        console.log("");
    */

    fs.writeFileSync(outputFileName, "" + matrixDimension + "\n", function () { });
    fs.appendFileSync(outputFileName, "" + dataEll.length + "," + colsEll.length + "\n", function () { });
    writeArrayToFile(dataEll, true);
    writeArrayToFile(colsEll, true);
    fs.appendFileSync(outputFileName, "" + dataCOO.length + "," + rowsCOO.length + "," + colsCOO.length + "\n", function () { });
    writeArrayToFile(dataCOO, true);
    writeArrayToFile(rowsCOO, true);
    writeArrayToFile(colsCOO, true);
    writeArrayToFile(vector, false);

    let end = new Date().getTime() / 1000;
    console.log("Done writing matrix and vector to file in " + roundNumber(end - start) + " seconds");
}

function roundNumber(number) {
    return Math.round((number + Number.EPSILON) * 10000) / 10000;
}

/**
 * 
 * @param {*} min 
 * @param {*} max 
 * @returns Returns a random number between min and max which is unequal to zero and is a decimal (not a whole number)
 */
function getRandomNumberInRange(min, max) {
    let random = 0;
    do {
        random = Math.random();
    } while (random === 0);
    let result = 0;
    do {
        result = (random * (max - min) + min);
    } while (result % 1 === 0);
    return result;
}

function writeArrayToFile(array, addNewLine) {
    let array_string = "";
    for (let i = 0; i < array.length; i++) {
        array_string += "," + array[i];
    }
    array_string = array_string.substring(1, array_string.length);
    if (addNewLine) {
        array_string += "\n";
    }
    fs.appendFileSync(outputFileName, array_string, function (err) { })
}