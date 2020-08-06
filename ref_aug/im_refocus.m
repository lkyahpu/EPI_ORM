clear vars;
addpath('function')
scene={'backgammon','pyramids','stripes','dots','boxes','cotton','dino','sideboard',...
             'antinous', 'boardgames', 'dishes',   'greek',...
             'kitchen',  'medieval2',  'museum',   'pens', ...   
             'pillows',  'platonic',   'rosemary', 'table', ...
             'tomb',     'tower',      'town',     'vinyl'};

stop=1.2;
begin=-1.7;
parameter.depth_resolution=10;
v_begin=-4;v_end=4;u_begin=-4;u_end=4;
xRes = 512;
yRes = 512;
uRes = 9;
vRes = 9;


for n=23
    parameter.name=scene{n};
    %output_path1 = ['G:/LKY/专利/code_matlab/',parameter.name,'/'];
    %mkdir(output_path1);
    
    output_path = ['G:/LKY/专利/code_matlab/',parameter.name,'/','refocus_5','/'];
    mkdir(output_path);
    
    name=parameter.name;
    % if strcmp(name,'pyramids');
    %     begin=-1.7; stop=1.2;
    % end
    % if strcmp(name,'backgammon');
    %     begin=-1.7; stop=0.7;
    % end
    % if strcmp(name,'stripes');
    %     begin=-0.3; stop=0.6;
    % end
    % if strcmp(name,'dots');
    %     begin=-0.7; stop=0.9;
    % end
    % if strcmp(name,'boxes');
    %     begin=-2.2; stop=1.4;
    % end
    % if strcmp(name,'cotton');
    %     begin=-1.6; stop=1.5;
    % end
    % if strcmp(name,'dino');
    %     begin=-1.9; stop=1.9;
    % end
    % if strcmp(name,'sideboard');
    %     begin=-2.0; stop=1.7;
    % end

    switch  name
        case 'pyramids'
            begin=-1.7; stop=1.2;
        case 'backgammon'
            begin=-1.7; stop=0.7;
        case 'stripes'
            begin=-0.3; stop=0.6;
        case 'dots'
            begin=-0.7; stop=0.9;
        case 'boxes'
            begin=-2.2; stop=1.4;
        case 'cotton'
            begin=-1.6; stop=1.5;
        case 'dino'
            begin=-1.9; stop=1.9;
        case 'sideboard'
            begin=-2.0; stop=1.7;
        case 'antinous'
            begin=-3.3; stop=2.8;
        case 'boardgames'
            begin=-1.8; stop=1.6;
        case 'dishes'
            begin=-3.1; stop=3.5;
        case 'greek'
            begin=-3.5; stop=3.1;
        case 'kitchen'
            begin=-1.6; stop=1.8;
        case 'medieval2'
            begin=-1.7; stop=2.0;
        case 'museum'
            begin=-1.5; stop=1.3;
        case 'pens'
            begin=-1.7; stop=2.0;
        case 'pillows'
            begin=-1.7; stop=1.8;
        case 'platonic'
            begin=-1.7; stop=1.5;
        case 'rosemary'
            begin=-1.8; stop=1.8;
        case 'table'
            begin=-2.0; stop=1.6;
        case 'tomb'
            begin=-1.5; stop=1.9;
        case 'tower'
            begin=-3.6; stop=3.5;
        case 'town'
            begin=-1.6; stop=1.6;
        case 'vinyl'
            begin=-1.6; stop=1.2;

    end

    depth_resolution =parameter.depth_resolution;

    
    input_string = strcat([name,'/'],strcat(name,'.png'));    % input path

    begin_temp=-stop;
    stop_temp=-begin;
    delta=(stop_temp-begin_temp)/(depth_resolution-1);

    UV_diameter = uRes;
    UV_center = round(UV_diameter/2);
    UV_radius = UV_center - 1;
    LF_y_size = yRes * vRes;
    LF_x_size = xRes * uRes;

    LF_Remap=double(imread(input_string));
    LF_Remap_alpha=zeros(LF_y_size,LF_x_size,3);

    for index=10:10
        alpha = stop_temp+delta-index*delta;
        %alpha = 0.3;
        IM_Refoc_alpha = zeros(yRes,xRes,3);
        refocus(double(xRes),double(yRes),...
            double(UV_diameter),double(UV_radius),LF_Remap,...
            LF_Remap_alpha,IM_Refoc_alpha,double(alpha),v_begin,v_end,u_begin,u_end);

        %imwrite(IM_Refoc_alpha,strcat(output_path,[name,'_refocus_',num2str(alpha),'.png']));
        %centr_Refoc_alpha=im2double(pinhole_img(LF_Remap_alpha,yRes,xRes,5,5));

        imwrite(LF_Remap_alpha/255,strcat(output_path,[name,'_ref_',num2str(double(alpha)),'_','.png']));
        %imwrite(LF_Remap_alpha/255,strcat(output_path,[name,'_ref_','06','.png']));
    end
    disp([scene{n},' ','finished'])
    
end
