const fs = require('fs')

const outputFileName = "matrix.csv";
const matrixDimension = 4;
const minValueDiagonalElement = 100;
const maxValueDiagonalElement = 200;
const maxValueNonDiagonalElement = 10;
let NUMBER_OF_NON_ZERO_ENTRIES_PER_ROW = 2;

let numberOfZeroes = 0;

const matrix = Array.from(Array(matrixDimension), () => new Array(matrixDimension));
const vector = Array(matrixDimension);

generateMatrix();
generateVector();
//printVectorAndMatrix();
//testMatrixAndVector();
//writeMatrixAndVectorToFile();
writeMatrixAndVectorToFileCsr();

function generateMatrix() {
    let start = new Date().getTime() / 1000;
    for (let i = 0; i < matrixDimension; i++) {

        let indicies = generateRandomIndicies(i);
        let currentIndex = indicies.pop();

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

function generateRandomIndicies(excluded) {

    let indicies = [];

    for (let i = 0; i < NUMBER_OF_NON_ZERO_ENTRIES_PER_ROW; i++) {
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

    let vectorRow = "";
    for (let i = 0; i < matrixDimension; i++) {
        vectorRow += "," + vector[i];
    }
    vectorRow = vectorRow.substring(1, vectorRow.length);
    fs.appendFileSync(outputFileName, vectorRow, function (err) { })

    let end = new Date().getTime() / 1000;
    console.log("Done writing matrix and vector to file in " + roundNumber(end - start) + " seconds");
}

function writeMatrixAndVectorToFileCsr() {
    let v = [];
    let col_index = [];
    let row_index = [];

    let start = new Date().getTime() / 1000;
    fs.writeFileSync(outputFileName, "" + matrixDimension + "\n", function () { });

    for (let i = 0; i < matrixDimension; i++) {
        let numberOfZeroesInCurrentRow = 0;
        for (let j = 0; j < matrixDimension; j++) {
            if (matrix[i][j] !== 0) {
                v.push(matrix[i][j]);
                col_index.push(j);
                numberOfZeroesInCurrentRow++;
            }
        }
        row_index.push(numberOfZeroesInCurrentRow);
    }

    let v_string = "";
    for (let i = 0; i < v.length; i++) {
        v_string += "," + v[i];
    }
    v_string = v_string.substring(1, v_string.length) + "\n";

    let col_index_string = "";
    for (let i = 0; i < col_index.length; i++) {
        col_index_string += "," + col_index[i];
    }
    col_index_string = col_index_string.substring(1, col_index_string.length) + "\n";

    let row_index_string = "";
    for (let i = 0; i < row_index.length; i++) {
        row_index_string += "," + row_index[i];
    }
    row_index_string = row_index_string.substring(1, row_index_string.length) + "\n";

    let vectorRow = "";
    for (let i = 0; i < matrixDimension; i++) {
        vectorRow += "," + vector[i];
    }
    vectorRow = vectorRow.substring(1, vectorRow.length);

    fs.appendFileSync(outputFileName, v_string, function (err) { })
    fs.appendFileSync(outputFileName, col_index_string, function (err) { })
    fs.appendFileSync(outputFileName, row_index_string, function (err) { })
    fs.appendFileSync(outputFileName, vectorRow, function (err) { })

    let end = new Date().getTime() / 1000;
    console.log("Done writing matrix and vector to file in " + roundNumber(end - start) + " seconds");
}

function roundNumber(number) {
    return Math.round((number + Number.EPSILON) * 10000) / 10000;
}

function getRandomNumberInRange(min, max) {
    return (Math.random() * (max - min) + min) + 0.0001;
}