features = {'dia'};
dist_metrics = {'l2'};

sets = {'train', 'test'};
datasets = ['espgame'];

test_image_count = [2081];
train_image_count = [18689];
dict_size = [268];

ids = 1;  

labels_per_image = 5;           %labels to be allotted per test image
nearest_neighbours = 5;         %number of nearest neighbours considered per test image


data_features = cell(numel(sets),train_image_count(ids));

fileId = fopen('espgame/espgame_data_vggf_pca_train.txt','r');

for i = 1:train_image_count(ids)
        data_features{1,i}=fscanf(fileId,'%f,');
end
fclose(fileId);

fileId = fopen('espgame/espgame_data_vggf_pca_test.txt','r');

for i = 1:test_image_count(ids)
        data_features{2,i} = fscanf(fileId,'%f,');
end
fclose(fileId);

distances = zeros(test_image_count(ids), train_image_count(ids));
for i = 1:test_image_count(ids)

    for j = 1:train_image_count(ids)
        dist = zeros(1, numel(features));
        for k = 1:numel(features)
            test_ft = data_features{2, i};
            train_ft = data_features{1, j};
            switch dist_metrics{k}
                case 'chi_square'
                    train_ft = train_ft / (0.0000000001+norm(train_ft));
                    test_ft = test_ft / (0.0000000001+norm(test_ft));
                    dist(k) = (0.5*sum(((train_ft-test_ft).^2)./(train_ft+test_ft+0.0000000001)));
                case 'l1'
                    train_ft = train_ft / (0.0000000001+sum(abs(train_ft)));
                    test_ft = test_ft / (0.0000000001+sum(abs(test_ft)));
                    dist(k) = sum(abs(train_ft-test_ft));
                case 'l2'
                    train_ft = train_ft / (0.0000000001+norm(train_ft));
                    test_ft = test_ft / (0.0000000001+norm(test_ft));
                    dist(k) = sqrt(sum((train_ft-test_ft).^2));
              end
        end
        distances(i, j) = sum(dist)/numel(features);
    end
end

save('espgame_dist.mat', 'distances', '-v7');
