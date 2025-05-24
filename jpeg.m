%% STEP 1: Read Image & Grayscale Conversion
image = imread('D:\University\Semnan University B.Sc. (Bachelor of Science)\8th Term\Lab - Digital Telecommunication\project_DT_LAB\HUFFMAN\Real_man\doggo.jpg');
imshow(image);
grayscale_image = 0.299*image(:,:,1) + 0.587*image(:,:,2) + 0.114*image(:,:,3);
imshow(grayscale_image);
[rows, columns] = size(grayscale_image);

%% STEP 2: Prepare Quantization Matrix
Q50 = [16 11 10 16 24 40 51 61;
       12 12 14 19 26 58 60 55;
       14 13 16 24 40 57 69 56;
       14 17 22 29 51 87 80 62;
       18 22 37 56 68 109 103 77;
       24 35 55 64 81 104 113 92;
       49 64 78 87 103 121 120 101;
       72 92 95 98 112 100 103 99];

%% STEP 3: Compression
block_size = 8;
all_symbols = {};
dc_values = [];
rle_lengths = [];

zigzag_map = zigzag_indices();

for i = 1:block_size:rows
    for j = 1:block_size:columns
        current_block = grayscale_image(i:i+7, j:j+7);


         if i == 1 && j == 1
            disp(current_block);
        end


        % DCT and Quantization
        dct_block = dct2(current_block);

        if i == 1 && j == 1
            disp(dct_block);
        end

        quantized_block = round(dct_block ./ Q50);


        if i == 1 && j == 1
            disp(quantized_block);  % Show ONLY the first 8x8 block
        end


        % Zigzag Scan
        flat_vector = zeros(1,64);
        for x = 1:8
            for y = 1:8
                flat_vector(zigzag_map(x,y)) = quantized_block(x,y);
            end
        end


         if i == 1 && j == 1
            disp(flat_vector);
         end


        % RLE Encoding (excluding DC)
        dc = flat_vector(1);
        ac = flat_vector(2:end);
        rle = jpegRLE(ac);

        % Symbol Generation for Huffman
        block_symbols = cell(size(rle,1),1);
        for k = 1:size(rle,1)
            block_symbols{k} = sprintf('%d_%d', rle(k,1), rle(k,2));
        end

         if i == 1 && j == 1
            disp(block_symbols);
         end

        % Store values for Huffman encoding
        all_symbols = [all_symbols; block_symbols];
        dc_values(end+1) = dc;
        rle_lengths(end+1) = size(rle,1);
    end
end

%% STEP 4: Huffman Encoding (globally once)
[unique_syms, ~, idx] = unique(all_symbols);
freqs = histcounts(idx, 1:numel(unique_syms)+1);
probs = freqs / sum(freqs);
dict = huffmandict(unique_syms, probs);
encoded_stream = huffmanenco(all_symbols, dict);

%% STEP 5: Huffman Decoding (globally once)
decoded_symbols = huffmandeco(encoded_stream, dict);
decoded_symbols = cellfun(@(s) strrep(s, ' ', '_'), decoded_symbols, 'UniformOutput', false);
symbol_pairs = cellfun(@(s) sscanf(s, '%d_%d'), decoded_symbols, 'UniformOutput', false);
pairs_mat = reshape(cell2mat(symbol_pairs), 2, []).';

%% STEP 6: Decompression
reconstructed_image = zeros(rows, columns);
start_idx = 1;
block_num = 0;

for i = 1:block_size:rows
    for j = 1:block_size:columns
        block_num = block_num + 1;
        
        % Retrieve correct RLE pairs
        len = rle_lengths(block_num);
        block_pairs = pairs_mat(start_idx:start_idx+len-1,:);
        start_idx = start_idx + len;

        % Reverse RLE
        ac_reconstructed = jpegInverseRLE(block_pairs);
        full_vector = [dc_values(block_num), ac_reconstructed];

        % Inverse Zigzag
        block_reconstructed = inverseZigzag(full_vector);

        % Inverse Quantization
        dequantized_block = block_reconstructed .* Q50;

        % Inverse DCT
        reconstructed_block = idct2(dequantized_block);

        % Store reconstructed block
        reconstructed_image(i:i+7, j:j+7) = reconstructed_block;
    end
end

imshow(uint8(reconstructed_image));
title('Fully Reconstructed Grayscale Image');

%% --- Supporting Functions ---

% Zigzag index mapping

function zigzag_map = zigzag_indices()
    zigzag_map = zeros(8);
    idx = 1;
    for s = 1:15
        if mod(s,2)==0
            for i = 1:8
                j = s-i+1;
                if i>=1 && i<=8 && j>=1 && j<=8
                    zigzag_map(i,j)=idx;
                    idx=idx+1;
                end
            end
        else
            for j=1:8
                i=s-j+1;
                if i>=1 && i<=8 && j>=1 && j<=8
                    zigzag_map(i,j)=idx;
                    idx=idx+1;
                end
            end
        end
    end
end

% JPEG RLE encoding
function rle = jpegRLE(ac_vector)
    rle = [];
    run_length = 0;
    for val = ac_vector
        if val==0
            run_length = run_length + 1;
            if run_length==16
                rle=[rle; 15,0];
                run_length=0;
            end
        else
            rle=[rle; run_length,val];
            run_length=0;
        end
    end
    if run_length>0
        rle=[rle; 0,0];
    end
end

% JPEG inverse RLE decoding
function ac_vector = jpegInverseRLE(rle)
    ac_vector=[];
    for k=1:size(rle,1)
        run=rle(k,1); val=rle(k,2);
        if run==0 && val==0
            ac_vector=[ac_vector, zeros(1,63-numel(ac_vector))];
            break;
        else
            ac_vector=[ac_vector, zeros(1,run), val];
        end
    end
end

% JPEG inverse zigzag
function block = inverseZigzag(vec)
    zigzag_map=zigzag_indices();
    block=zeros(8);
    for x=1:8
        for y=1:8
            block(x,y)=vec(zigzag_map(x,y));
        end
    end
end

